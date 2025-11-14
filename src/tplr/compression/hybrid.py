import math
from typing import Dict
from typing import List, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

BytesLike = Union[bytes, bytearray, np.ndarray, torch.Tensor]


def encode_batch_rows_sorted(
        idx_sorted: torch.Tensor,
        *,
        C: int,
        B_choices: Tuple[int, ...] = (64, 128)
) -> Tuple[BytesLike, Dict]:
    """
    Compresses a 2D tensor of Top-K indices into a byte string
    using a per-row adaptive Rice/Bitmap compression scheme on the GPU.

    Args:
        idx_sorted (torch.Tensor): [rows, k] sorted tensor of indices.
        C (int): The total number of columns (0 <= idx < C).
        B_choices (tuple[int, ...]): Block sizes to evaluate.
                                     Must be powers of two.
                                     Must evenly divide C.

    Returns:
        tuple[bytes, dict]: (payload, meta)
            - payload (bytes): The compressed byte string.
            - meta (dict): Metadata about the compression.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this function.")

    if not isinstance(idx_sorted, torch.Tensor) or idx_sorted.ndim != 2:
        raise ValueError(f"idx must be a 2D int64 tensor, got {idx_sorted.shape}")

    if not all(isinstance(b, int) and (b & (b - 1) == 0) and b > 0 for b in B_choices):
        raise ValueError(f"All B_choices must be powers of two, got {B_choices}")

    if not all(C % b == 0 for b in B_choices):
        raise ValueError(f"All B_choices must evenly divide C={C}, got {B_choices}")

    num_rows, k_dim = idx_sorted.shape
    if num_rows == 0:
        return b"", {
            "total_bits": 0,
            "avg_bits_per_row": 0.0,
            "B_hist": {b: 0 for b in B_choices}
        }

    if not idx_sorted.is_cuda:
        idx_sorted = idx_sorted.cuda()
    idx_sorted = idx_sorted.contiguous()
    dev = idx_sorted.device

    # Calculate k_rice parameters (log2(C // B))
    k_rice_choices = tuple(int(math.log2(C // b)) for b in B_choices)
    num_B_choices = len(B_choices)
    k_rice_choices_tensor = torch.tensor(k_rice_choices, dtype=torch.int32, device=dev)
    B_choice_bits = (num_B_choices - 1).bit_length()

    # Row header: 1 bit (bitmap/rice) + B_choice_bits
    ROW_HEADER_BITS = 1 + B_choice_bits

    # Delta encode: val[0], val[1]-val[0], val[2]-val[1], ...
    delta = torch.cat(
        (idx_sorted[:, :1], idx_sorted[:, 1:] - idx_sorted[:, :-1]),
        dim=1
    )

    # Output tensors for cost kernel
    costs = torch.empty((num_rows, num_B_choices), dtype=torch.int32, device=dev)
    is_bitmap = torch.empty((num_rows, num_B_choices), dtype=torch.int8, device=dev)
    grid = (num_rows,)

    # k_dim is passed as constexpr for tl.arange, but B_choices are dynamic
    cost_kernel[grid](
        delta,
        costs,
        is_bitmap,
        k_dim=k_dim,
        num_rows=num_rows,
        num_B_choices=num_B_choices,
        k_rice_choices_ptr=k_rice_choices_tensor,
    )

    # pick best B/mode & compute layout

    # Best choice per row
    min_costs, best_B_idx = torch.min(costs, dim=1)
    is_bitmap_choice = torch.gather(is_bitmap, 1, best_B_idx.unsqueeze(1)).squeeze(1).to(torch.int32)

    # (1) payload bits per row (deltas only)
    row_payload_bits = min_costs + ROW_HEADER_BITS  # (rows,)

    # (2) payload bytes per row (rounded up)
    row_payload_bytes = ((row_payload_bits + 7) // 8).to(torch.int32)  # (rows,)

    # (3) on-wire bits per row = 16 (length) + payload rounded to bytes
    row_bits_aligned = (16 + row_payload_bytes * 8).to(torch.int64)  # (rows,)

    # (4) starting bit offsets (before header)
    row_bit_offsets = torch.nn.functional.pad(
        torch.cumsum(row_bits_aligned, dim=0, dtype=torch.int64)[:-1],
        (1, 0)
    )

    # (5) total bits across all rows (Python int)
    total_bits = int(row_bits_aligned.sum().item())

    # Build global header bytes
    header_list = []
    header_list.append(b"CGRP")  # 4B magic
    header_list.append(int(C).to_bytes(4, "little"))  # 4B C (uint32 LE)
    header_list.append(int(k_dim).to_bytes(2, "little"))  # 2B K (uint16 LE)  <--- NEW
    header_list.append(bytes([len(B_choices)]))  # 1B num_B
    for b in B_choices:
        header_list.append(int(b).to_bytes(2, "little"))  # 2B per B (uint16 LE)

    global_header_py = b"".join(header_list)
    global_header_len_bytes = len(global_header_py)

    # shift row starts by header
    row_bit_offsets = row_bit_offsets + global_header_len_bytes * 8

    # final sizes (Python ints)
    total_payload_bytes = (total_bits + 7) // 8
    final_buffer_bytes = global_header_len_bytes + total_payload_bytes

    # allocate + write header
    payload_buf = torch.zeros(final_buffer_bytes, dtype=torch.uint8, device=dev)
    payload_buf[:global_header_len_bytes] = torch.tensor(
        list(global_header_py), dtype=torch.uint8, device=dev
    )

    pack_kernel[(num_rows,)](
        delta,
        payload_buf,
        row_bit_offsets.to(torch.int32),
        row_payload_bytes,  # already int32
        best_B_idx.to(torch.int32),
        is_bitmap_choice,  # int32 0/1
        k_rice_choices_tensor,
        num_rows,
        k_dim=k_dim,
        ROW_HEADER_BITS=ROW_HEADER_BITS,
    )

    payload_cpu = payload_buf.cpu()
    b_counts = torch.bincount(best_B_idx, minlength=len(B_choices))
    B_hist = {b: c.item() for b, c in zip(B_choices, b_counts)}
    meta = {
        "total_bits": total_bits,  # includes 16-bit length and byte padding
        "avg_bits_per_row": float(row_bits_aligned.float().mean().item()),
        "avg_payload_bits_per_row": float(row_payload_bits.float().mean().item()),
        "B_hist": B_hist,
    }
    return payload_cpu, meta


@triton.jit
def cost_kernel(
        delta_ptr,              # (rows, k_dim) IN
        costs_ptr,              # (rows, num_B_choices) OUT
        is_bitmap_ptr,          # (rows, num_B_choices) OUT (bool/int)
        k_dim: tl.constexpr,    # constexpr for tl.arange
        num_rows: tl.int32,
        num_B_choices: tl.int32,
        k_rice_choices_ptr,     # (num_B_choices,) int32
):
    """
    Calculates the compressed bit cost for each row for each B in B_choices.
    One program instance processes one row.
    Variant B: first delta encoded with Rice, tail optionally bitmap (q in {0,1}).
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    # Lane indices for this row (constexpr width)
    i = tl.arange(0, k_dim)

    # Load the entire row of delta-encoded values into SRAM
    row_base = row_idx * k_dim
    delta = tl.load(delta_ptr + row_base + i)
    delta0 = tl.load(delta_ptr + row_base)

    b_idx = 0
    while b_idx < num_B_choices:
        # k_rice and M = 1 << k_rice
        k_rice = tl.load(k_rice_choices_ptr + b_idx)

        # q via shift, r via mask
        q = delta >> k_rice
        q0 = delta0 >> k_rice

        # Pure Rice cost: sum(unary(q)) + sum(r)   where unary(q) has (q + 1) bits,
        # and r contributes k_rice bits per element.
        rice_cost = tl.sum(q + 1) + k_dim * k_rice

        # Bitmap cost: first element full Rice, tail has (1 + k_rice) bits
        #   (1 bit for q in {0,1} + k_rice bits for r)
        bitmap_cost = (q0 + 1 + k_rice) + (k_dim - 1) * (1 + k_rice)
        # equivalently: bitmap_cost = k_dim * (1 + k_rice) + q0

        # Allow bitmap only if tail q are in {0,1}
        # Compute tail max with a masked reduction (ignore lane 0)
        q_tail_max = tl.max(tl.where(i > 0, q, 0))
        bitmap_allowed = q_tail_max <= 1

        use_bitmap = (bitmap_cost < rice_cost) & bitmap_allowed
        min_cost = tl.where(use_bitmap, bitmap_cost, rice_cost)

        out_offset = row_idx * num_B_choices + b_idx
        tl.store(costs_ptr + out_offset, min_cost)
        # make sure is_bitmap is exactly 0/1 in memory
        tl.store(
            is_bitmap_ptr + out_offset,
            tl.where(use_bitmap, 1, 0).to(tl.int32),
        )
        b_idx += 1


@triton.jit
def write_nbits(u8_ptr, bit_off_i32, value_u32, nbits_i32):
    """
    Write `nbits_i32` bits from `value_u32` at bit offset `bit_off_i32` (LSB-first).
    All args are Triton scalars:
      - bit_off_i32 : tl.int32
      - value_u32   : tl.uint32 (<= 32 bits)
      - nbits_i32   : tl.int32
    Returns new bit offset (tl.int32).
    """
    j = tl.full((), 0, dtype=tl.int32)
    ONE_U32 = tl.full((), 1, dtype=tl.uint32)

    while j < nbits_i32:
        pos = bit_off_i32 + j
        byte_idx = (pos >> 3).to(tl.int32)
        bit_idx = (pos & 7).to(tl.int32)

        old_u8 = tl.load(u8_ptr + byte_idx)
        old_u32 = old_u8.to(tl.uint32)

        vbit = (value_u32 >> j) & ONE_U32
        mask = ONE_U32 << bit_idx
        new_u32 = (old_u32 & (~mask)) | (vbit << bit_idx)
        tl.store(u8_ptr + byte_idx, new_u32.to(tl.uint8))
        j += 1
    return bit_off_i32 + nbits_i32


@triton.jit
def pack_kernel(
    delta_ptr,              # (rows, k_dim) IN int32
    u8_payload_ptr,         # (final_buffer_bytes,) OUT uint8
    row_bit_offsets_ptr,    # (rows,) IN  (int32 preferred)
    row_payload_bytes_ptr,  # (rows,) IN  int32
    best_B_idx_ptr,         # (rows,) IN  int32
    is_bitmap_ptr,          # (rows,) IN  int32 (0/1)
    k_rice_choices_ptr,     # [num_B] IN int32
    num_rows: tl.int32,
    k_dim: tl.int32,        # dynamic
    ROW_HEADER_BITS: tl.constexpr,
):
    """
    Variant B: first delta Rice (unary = q ones then 0) + r; tail bitmap or Rice.
    Bit order: LSB-first.
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    # per-row meta
    bit_off_i32 = tl.load(row_bit_offsets_ptr + row_idx).to(tl.int32)
    payload_bytes_i32 = tl.load(row_payload_bytes_ptr + row_idx).to(tl.int32)
    b_idx_i32 = tl.load(best_B_idx_ptr + row_idx).to(tl.int32)
    use_bitmap_i32 = (tl.load(is_bitmap_ptr + row_idx) & 1).to(tl.int32)

    # params
    k_rice_i32 = tl.load(k_rice_choices_ptr + b_idx_i32).to(tl.int32)
    M_i32 = (tl.full((), 1, dtype=tl.int32) << k_rice_i32)

    ONE_U32 = tl.full((), 1, dtype=tl.uint32)
    ZERO_U32 = tl.full((), 0, dtype=tl.uint32)
    ONE_I32 = tl.full((), 1, dtype=tl.int32)
    THIRTY_ONE_I32 = tl.full((), 31, dtype=tl.int32)

    # 16-bit length
    bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32,
                              payload_bytes_i32.to(tl.uint32),
                              tl.full((), 16, dtype=tl.int32))

    # header ((b_idx << 1) | use_bitmap)
    header_i32 = (b_idx_i32 << 1) | use_bitmap_i32
    bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32,
                              header_i32.to(tl.uint32),
                              tl.full((), ROW_HEADER_BITS, dtype=tl.int32))

    base = row_idx * k_dim

    # ---- first delta: ALWAYS Rice ----
    if k_dim > 0:
        v0 = tl.load(delta_ptr + base).to(tl.int32)
        q0 = (v0 >> k_rice_i32).to(tl.int32)
        r0 = (v0 & (M_i32 - 1)).to(tl.int32)

        # q0 ones in chunks of <=31, then a single 0
        q_left = q0
        while q_left > 0:
            chunk = tl.minimum(q_left, THIRTY_ONE_I32)
            ones  = (ONE_U32 << chunk) - ONE_U32
            bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, ones, chunk)
            q_left -= chunk
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, ZERO_U32, ONE_I32) # terminating 0
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, r0.to(tl.uint32), k_rice_i32) # remainder

    # tail deltas
    i = 1
    while i < k_dim:
        v = tl.load(delta_ptr + base + i).to(tl.int32)
        q = (v >> k_rice_i32).to(tl.int32)
        r = (v & (M_i32 - 1)).to(tl.int32)

        # Rice unary only if NOT bitmap
        q_left = tl.where(use_bitmap_i32 != 0, tl.full((), 0, dtype=tl.int32), q)
        while q_left > 0:
            chunk = tl.minimum(q_left, THIRTY_ONE_I32)
            ones  = (ONE_U32 << chunk) - ONE_U32
            bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, ones, chunk)
            q_left -= chunk
        n_term = tl.where(use_bitmap_i32 != 0, tl.full((), 0, dtype=tl.int32), ONE_I32)
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, ZERO_U32, n_term)

        # bitmap q only if bitmap
        q_bit  = tl.where(q > 0, ONE_U32, ZERO_U32)
        n_qbit = tl.where(use_bitmap_i32 != 0, ONE_I32, tl.full((), 0, dtype=tl.int32))
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, q_bit, n_qbit)

        # remainder always
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, r.to(tl.uint32), k_rice_i32)
        i += 1


class BitStreamReader:
    """
    LSB-first bit reader over a bytes-like buffer (torch.uint8, np.uint8, or Python bytes).
    - read_bits(n): reads n bits, returning an integer whose bit j is the j-th bit read.
    - read_unary_bounded(end_bit): reads '1's until a '0' or end_bit; returns (q, hit_end)
    """
    __slots__ = ("buf", "total_bits", "bit_off")

    def __init__(self, payload: BytesLike, bit_offset_start: int = 0):
        if isinstance(payload, torch.Tensor):
            assert payload.dtype == torch.uint8
            self.buf = payload.cpu().numpy().tobytes()
        elif isinstance(payload, np.ndarray):
            assert payload.dtype == np.uint8
            self.buf = payload.tobytes()
        elif isinstance(payload, (bytes, bytearray)):
            self.buf = bytes(payload)
        else:
            raise TypeError("Unsupported payload type for BitStreamReader")

        self.total_bits = len(self.buf) * 8
        self.bit_off = int(bit_offset_start)

    def read_bits(self, n_bits: int) -> int:
        """Read n_bits in LSB-first order; returns value with bit j equal to j-th bit read."""
        if n_bits == 0:
            return 0
        if self.bit_off + n_bits > self.total_bits:
            raise EOFError("Attempt to read past end of bitstream")
        val = 0
        start = self.bit_off
        for j in range(n_bits):
            pos = start + j
            b = self.buf[pos >> 3]
            bit = (b >> (pos & 7)) & 1
            val |= (bit << j)
        self.bit_off = start + n_bits
        return val

    def read_unary_bounded(self, end_bit: int) -> Tuple[int, bool]:
        """
        Read unary as q ones followed by a single 0, *bounded* by end_bit.
        Returns: (q, hit_end)
          - q: number of 1s seen before the terminating 0
          - hit_end: True if we reached end_bit before seeing the terminating 0
        """
        q = 0
        while self.bit_off < end_bit:
            bit = self.read_bits(1)
            if bit == 1:
                q += 1
            else:
                return q, False
        return q, True  # ran out of bits without seeing the terminating 0

    def bits_remaining(self) -> int:
        return self.total_bits - self.bit_off

    def is_at_end(self) -> bool:
        """True if only up to 7 padding bits remain globally."""
        return self.bit_off >= self.total_bits - 7


def _parse_global_header(payload: BytesLike) -> Tuple[int, int, list[int], int]:
    """
    Layout:
      4B "CGRP"
      4B C (uint32 LE)
      2B K (uint16 LE)
      1B num_B
      2B * num_B  (each B, uint16 LE)
    Returns: (C, K, B_choices, header_end_bit_offset)
    """
    if isinstance(payload, torch.Tensor):
        assert payload.dtype == torch.uint8
        raw = payload.cpu().numpy().tobytes()
    elif isinstance(payload, np.ndarray):
        assert payload.dtype == np.uint8
        raw = payload.tobytes()
    elif isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    else:
        raise TypeError("Unsupported payload type")

    if len(raw) < 11:
        raise ValueError("Payload too short for global header")
    if raw[:4] != b"CGRP":
        raise ValueError("Bad magic; expected 'CGRP'")

    C     = int.from_bytes(raw[4:8],  "little", signed=False)
    K     = int.from_bytes(raw[8:10], "little", signed=False)
    num_B = raw[10]
    need  = 4 + 4 + 2 + 1 + 2 * num_B
    if len(raw) < need:
        raise ValueError("Payload shorter than header requires")

    B_choices = []
    off = 11
    for _ in range(num_B):
        b = int.from_bytes(raw[off:off+2], "little", signed=False)
        B_choices.append(b)
        off += 2
    return C, K, B_choices, off * 8


def _decode_row(stream: BitStreamReader, M: int, k_rice: int, use_bitmap: int,
                row_payload_bytes: int, row_header_bits: int,
                K: int) -> list[int]:
    """
    Stream is positioned just AFTER the 16-bit length.
    Decode EXACTLY K deltas (first is Rice; tail is bitmap or Rice),
    then align to end-of-row (row_payload_bytes*8).
    """
    start_bit   = stream.bit_off
    row_end_bit = start_bit + row_payload_bytes * 8

    # header
    _ = stream.read_bits(row_header_bits)

    deltas: list[int] = []

    # first (Rice)
    q0, hit_end = stream.read_unary_bounded(row_end_bit)
    if hit_end or stream.bit_off + k_rice > row_end_bit:
        stream.bit_off = row_end_bit
        return []
    r0 = stream.read_bits(k_rice)
    deltas.append(q0 * M + r0)

    # Tail: exactly K-1 more
    for _ in range(K - 1):
        if use_bitmap:
            need = 1 + k_rice
            if stream.bit_off + need > row_end_bit:
                # not enough bits; treat as malformed/padded
                break
            q = stream.read_bits(1)
            r = stream.read_bits(k_rice)
        else:
            # Rice
            if stream.bit_off >= row_end_bit:
                break
            q, hit_end = stream.read_unary_bounded(row_end_bit)
            if hit_end or stream.bit_off + k_rice > row_end_bit:
                break
            r = stream.read_bits(k_rice)
        deltas.append(q * M + r)

    # align to end of row explicitly
    stream.bit_off = row_end_bit

    # prefix sum
    if not deltas:
        return []
    vals = [0] * len(deltas)
    vals[0] = deltas[0]
    for i in range(1, len(deltas)):
        vals[i] = vals[i-1] + deltas[i]
    return vals


def decode_batch_rows(payload: BytesLike) -> tuple[list[list[int]], int, int]:
    C, K, B_choices, header_end_bit = _parse_global_header(payload)
    num_B = len(B_choices)

    # derive M/k_rice per choice
    M_choices, k_rice_choices = [], []
    for B in B_choices:
        M = C // B
        if M <= 0 or (M & (M - 1)) != 0:
            raise ValueError(f"M=C//B={M} not power of two for B={B}")
        M_choices.append(M)
        k_rice_choices.append(int(math.log2(M)))

    B_choice_bits  = (num_B - 1).bit_length()
    ROW_HEADER_BITS = 1 + B_choice_bits

    stream = BitStreamReader(payload, bit_offset_start=header_end_bit)
    rows_out: list[list[int]] = []
    while stream.bits_remaining() >= 16:
        row_payload_bytes = stream.read_bits(16)
        if row_payload_bytes == 0 and stream.is_at_end():
            break

        # Peek header to learn best_B_idx & use_bitmap
        if stream.bits_remaining() < ROW_HEADER_BITS:
            break
        header = stream.read_bits(ROW_HEADER_BITS)
        use_bitmap = header & 1
        best_B_idx = header >> 1

        if not (0 <= best_B_idx < num_B):
            break
        M = M_choices[best_B_idx]
        k_rice = k_rice_choices[best_B_idx]

        # Rewind header; decode the row with exact K
        stream.bit_off -= ROW_HEADER_BITS
        row_vals = _decode_row(
            stream, M=M, k_rice=k_rice, use_bitmap=use_bitmap,
            row_payload_bytes=row_payload_bytes, row_header_bits=ROW_HEADER_BITS,
            K=K
        )
        if not row_vals:
            break
        rows_out.append(row_vals)
    return rows_out, C, len(rows_out)


if __name__ == "__main__":
    torch.manual_seed(0)
    ROWS, K = 32, 16
    COLS = 4096

    x = torch.randn((ROWS, COLS), dtype=torch.float32)
    idx = torch.topk(x.abs(), k=K, dim=-1, largest=True, sorted=False).indices

    idx, _ = torch.sort(idx, dim=1)
    payload, _ = encode_batch_rows_sorted(idx, C=COLS, B_choices=(64, 128, 256))
    decoded, _, _ = decode_batch_rows(payload)
    ok = True
    idx = [row for row in idx]
    for r in range(ROWS):
        if not torch.equal(torch.tensor(decoded[r]), idx[r].cpu()):
            ok = False
            print("Mismatch row", r)
            print("orig:", idx[r].tolist())
            print("dec :", decoded[r])
    print("Round-trip OK" if ok else "Round-trip MISMATCH")
