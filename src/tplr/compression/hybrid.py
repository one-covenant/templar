import math
from typing import Dict
from typing import Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

BytesLike = Union[bytes, bytearray, np.ndarray, torch.Tensor]


@torch.no_grad()
def encode_batch_rows(
        idx: torch.Tensor,
        *,
        C: int,
        use_delta: bool = True,
        B_choices: Tuple[int, ...] = (64, 128)
) -> Tuple[BytesLike, Dict]:
    """
    Compresses a 2D int64 tensor of Top-K indices into a byte string
    using a per-row adaptive Rice/Bitmap compression scheme on the GPU.

    Layout:
    0..3   : "CGRP"         (magic)
    4..7   : C      (uint32 LE)
    8..9   : K      (uint16 LE)
    10..13 : R      (uint32 LE, num_rows)
    14     : num_B  (uint8)
    15..   : B_choices (num_B * uint16 LE)

    Args:
        idx (torch.Tensor): [rows, k] int64 tensor of indices.
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

    if not isinstance(idx, torch.Tensor) or idx.ndim != 2:
        raise ValueError(f"idx must be a 2D int64 tensor, got {idx.shape} {idx.dtype}")

    if not all(isinstance(b, int) and (b & (b - 1) == 0) and b > 0 for b in B_choices):
        raise ValueError(f"All B_choices must be powers of two, got {B_choices}")

    if not all(C % b == 0 for b in B_choices):
        raise ValueError(f"All B_choices must evenly divide C={C}, got {B_choices}")

    num_rows, k_dim = idx.shape
    if num_rows == 0:
        return b"", {
            "total_bits": 0,
            "avg_bits_per_row": 0.0,
            "B_hist": {b: 0 for b in B_choices}
        }

    if not idx.is_cuda:
        idx = idx.cuda()
    idx = idx.contiguous()
    dev = idx.device

    if use_delta:
        # v[0], v[1]-v[0], v[2]-v[1], ...
        vals = torch.cat(
            (idx[:, :1], idx[:, 1:] - idx[:, :-1]),
            dim=1,
        )
    else:
        vals = idx

    # Cast to int32 for Triton kernels
    vals = vals.to(torch.int32)

    # Calculate k_rice parameters (log2(C // B))
    k_rice_choices = tuple(int(math.log2(C // b)) for b in B_choices)
    num_B_choices = len(B_choices)
    k_rice_choices_tensor = torch.tensor(k_rice_choices, dtype=torch.int32, device=dev)
    B_choice_bits = (num_B_choices - 1).bit_length()

    # Row header: 1 bit (bitmap/rice) + B_choice_bits
    ROW_HEADER_BITS = 1 + B_choice_bits

    # Output tensors for cost kernel
    costs = torch.empty((num_rows, num_B_choices), dtype=torch.int32, device=dev)
    is_bitmap = torch.empty((num_rows, num_B_choices), dtype=torch.int8, device=dev)
    grid = (num_rows,)

    # Launch cost kernel
    # k_dim is passed as constexpr for tl.arange, but B_choices are dynamic
    cost_kernel[grid](
        vals,
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
    header_list.append(int(C).to_bytes(4, "little"))       # 4B C (uint32 LE)
    header_list.append(int(k_dim).to_bytes(2, "little"))   # 2B K (uint16 LE)
    header_list.append(int(num_rows).to_bytes(4, "little"))  # 4B R (uint32 LE) NEW
    header_list.append(bytes([len(B_choices)]))            # 1B num_B
    for b in B_choices:
        header_list.append(int(b).to_bytes(2, "little"))   # 2B per B (uint16 LE)

    global_header_py = b"".join(header_list)
    global_header_len_bytes = len(global_header_py)
    # this is 15 + 2 * len(B_choices)

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
        vals,
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

    b_counts = torch.bincount(best_B_idx, minlength=len(B_choices))
    B_hist = {b: c.item() for b, c in zip(B_choices, b_counts)}
    meta = {
        "total_bits": total_bits,  # includes 16-bit length and byte padding
        "avg_bits_per_row": float(row_bits_aligned.float().mean().item()),
        "avg_payload_bits_per_row": float(row_payload_bits.float().mean().item()),
        # header+payload, no 16-bit length, before byte-rounding
        "B_hist": B_hist,
    }
    return payload_buf, meta


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

    # Load entire row of delta-encoded values into SRAM
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

        # Pure Rice cost: sum(q + 1) + k_dim * k_rice
        rice_cost = tl.sum(q + 1) + k_dim * k_rice

        # Bitmap cost: first element full Rice, tail has (1 + k_rice) bits
        bitmap_cost = (q0 + 1 + k_rice) + (k_dim - 1) * (1 + k_rice)

        # Allow bitmap only if tail q are in {0,1}
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
def write_nbits(
    u8_ptr, # uint8* global buffer
    bit_off_i32, # scalar tl.int32 bit offset
    value_u32, # scalar tl.uint32, up to 32 bits used
    nbits_i32, # scalar tl.int32, number of bits to write
):
    """
    Writes `nbits_i32` least-significant bits of `value_u32` into `u8_ptr`
    starting at bit offset `bit_off_i32` in LSB-first order.

    This is still a bit-at-a-time writer; higher-level kernels have been
    adjusted to use int32 + shift/mask ahead of time.
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
    First delta Rice (unary = q ones then 0) + r; tail bitmap or Rice.
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

    # first delta: ALWAYS Rice
    if k_dim > 0:
        v0 = tl.load(delta_ptr + base).to(tl.int32)
        q0 = (v0 >> k_rice_i32).to(tl.int32)
        r0 = (v0 & (M_i32 - 1)).to(tl.int32)

        # q0 ones in chunks of <=31, then a single 0
        q_left = q0
        while q_left > 0:
            chunk = tl.minimum(q_left, THIRTY_ONE_I32)
            ones = (ONE_U32 << chunk) - ONE_U32
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
            ones = (ONE_U32 << chunk) - ONE_U32
            bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, ones, chunk)
            q_left -= chunk
        n_term = tl.where(use_bitmap_i32 != 0, tl.full((), 0, dtype=tl.int32), ONE_I32)
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, ZERO_U32, n_term)

        # bitmap q only if bitmap
        q_bit = tl.where(q > 0, ONE_U32, ZERO_U32)
        n_qbit = tl.where(use_bitmap_i32 != 0, ONE_I32, tl.full((), 0, dtype=tl.int32))
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, q_bit, n_qbit)

        # remainder always
        bit_off_i32 = write_nbits(u8_payload_ptr, bit_off_i32, r.to(tl.uint32), k_rice_i32)
        i += 1


@triton.jit
def read_nbits_triton(u8_ptr, bit_off_i32, nbits_i32, limit_bit_i32):
    """
    GPU version of BitStreamReader.read_bits (LSB-first), but bounds-safe.

    Reads `nbits_i32` bits starting at `bit_off_i32`, but never loads beyond
    bit index `limit_bit_i32` (masked loads return 0 out-of-bounds).

    Returns: (value_u32, new_bit_off_i32)
    """
    j = tl.full((), 0, dtype=tl.int32)
    val_u32 = tl.full((), 0, dtype=tl.uint32)
    ONE_U32 = tl.full((), 1, dtype=tl.uint32)
    ZERO_U8 = tl.full((), 0, dtype=tl.uint8)

    while j < nbits_i32:
        pos = bit_off_i32 + j
        in_bounds = pos < limit_bit_i32

        byte_idx = (pos >> 3).to(tl.int32)
        bit_idx = (pos & 7).to(tl.int32)

        # Masked load: if in_bounds==0, we load ZERO_U8 instead of touching memory.
        u8 = tl.load(u8_ptr + byte_idx, mask=in_bounds, other=ZERO_U8)
        u32 = u8.to(tl.uint32)
        bit = (u32 >> bit_idx) & ONE_U32

        val_u32 |= (bit << j)
        j += 1

    new_bit_off = bit_off_i32 + nbits_i32
    return val_u32, new_bit_off


@triton.jit
def read_unary_bounded_triton(u8_ptr, bit_off_i32, end_bit_i32):
    """
    GPU version of BitStreamReader.read_unary_bounded(end_bit).
    Reads '1's until a '0' or end_bit.
    Returns: (q_i32, new_bit_off_i32, hit_end_i32)
      - q_i32: number of 1s before the terminating 0
      - hit_end_i32: 1 if we reached end_bit without seeing 0
                     0 if we saw a terminating 0
    """
    ONE_U32 = tl.full((), 1, dtype=tl.uint32)
    q_i32 = tl.full((), 0, dtype=tl.int32)
    hit_end_i32 = tl.full((), 1, dtype=tl.int32)

    cond = bit_off_i32 < end_bit_i32
    while cond:
        pos = bit_off_i32
        byte_idx = (pos >> 3).to(tl.int32)
        bit_idx = (pos & 7).to(tl.int32)

        u8 = tl.load(u8_ptr + byte_idx)
        u32 = u8.to(tl.uint32)
        bit = (u32 >> bit_idx) & ONE_U32

        bit_off_i32 += 1

        is_one = (bit == ONE_U32)
        q_i32 += is_one.to(tl.int32)

        # If bit is 0, we did NOT hit end
        hit_end_i32 = tl.where(is_one, hit_end_i32,
                               tl.full((), 0, dtype=tl.int32))

        # Continue only if we are still inside the row and last bit was 1
        cond = (bit_off_i32 < end_bit_i32) & is_one

    return q_i32, bit_off_i32, hit_end_i32


@triton.jit
def parse_header_kernel(
    u8_payload_ptr,           # (total_bytes,) uint8
    C_out_ptr,                # (1,) int32
    K_out_ptr,                # (1,) int32
    R_out_ptr,                # (1,) int32  NEW: num_rows
    num_B_out_ptr,            # (1,) int32
    B_choices_out_ptr,        # (MAX_B_CHOICES,) int32
    header_bytes_out_ptr,     # (1,) int32
    error_flag_ptr,           # (1,) int32
    total_bytes: tl.int32,
    MAX_B_CHOICES: tl.constexpr,
):
    """
    Parse the global header entirely on GPU.

    Layout:
      0..3   : "CGRP"
      4..7   : C (uint32 LE)
      8..9   : K (uint16 LE)
      10..13 : R (uint32 LE, num_rows)
      14     : num_B (uint8)
      15..   : B_choices (num_B * 2 bytes, uint16 LE)
    """

    pid = tl.program_id(0)
    if pid != 0:
        return

    # ---- init outputs / error ----
    C_val = tl.full((), 0, dtype=tl.int32)
    K_val = tl.full((), 0, dtype=tl.int32)
    R_val = tl.full((), 0, dtype=tl.int32)
    num_B_val = tl.full((), 0, dtype=tl.int32)
    header_bytes_i32 = tl.full((), 0, dtype=tl.int32)
    err = tl.full((), 0, dtype=tl.int32)

    # ---- basic size + magic checks ----
    # Minimum header size: 15 bytes (without B_choices)
    if total_bytes < 15:
        err = 1
    else:
        # Magic "CGRP" = [67, 71, 82, 80]
        m0 = tl.load(u8_payload_ptr + 0)
        m1 = tl.load(u8_payload_ptr + 1)
        m2 = tl.load(u8_payload_ptr + 2)
        m3 = tl.load(u8_payload_ptr + 3)
        cond_magic = (m0 == 67) & (m1 == 71) & (m2 == 82) & (m3 == 80)
        bad_magic = cond_magic == 0
        err = tl.where(bad_magic, tl.full((), 2, dtype=tl.int32), err)

    # ---- C, K, R, num_B ----
    if err == 0:
        # C (uint32 LE at bytes 4..7)
        b4 = tl.load(u8_payload_ptr + 4).to(tl.int32)
        b5 = tl.load(u8_payload_ptr + 5).to(tl.int32)
        b6 = tl.load(u8_payload_ptr + 6).to(tl.int32)
        b7 = tl.load(u8_payload_ptr + 7).to(tl.int32)
        C_val = b4 | (b5 << 8) | (b6 << 16) | (b7 << 24)

        # K (uint16 LE at bytes 8..9)
        b8 = tl.load(u8_payload_ptr + 8).to(tl.int32)
        b9 = tl.load(u8_payload_ptr + 9).to(tl.int32)
        K_val = b8 | (b9 << 8)

        # R (uint32 LE at bytes 10..13)
        b10 = tl.load(u8_payload_ptr + 10).to(tl.int32)
        b11 = tl.load(u8_payload_ptr + 11).to(tl.int32)
        b12 = tl.load(u8_payload_ptr + 12).to(tl.int32)
        b13 = tl.load(u8_payload_ptr + 13).to(tl.int32)
        R_val = b10 | (b11 << 8) | (b12 << 16) | (b13 << 24)

        # num_B at byte 14
        num_B_val = tl.load(u8_payload_ptr + 14).to(tl.int32)
        invalid_num_B = (num_B_val <= 0) | (num_B_val > MAX_B_CHOICES)
        err = tl.where(invalid_num_B, tl.full((), 3, dtype=tl.int32), err)

    # ---- read B_choices in a structured loop (no break/return) ----
    off = tl.full((), 15, dtype=tl.int32)  # B_choices start at byte 15
    i = tl.full((), 0, dtype=tl.int32)

    while i < MAX_B_CHOICES:
        need_this = (i < num_B_val) & (err == 0)

        if need_this:
            cond_in_bounds = (off + 1) < total_bytes
            if cond_in_bounds:
                lo = tl.load(u8_payload_ptr + off).to(tl.int32)
                hi = tl.load(u8_payload_ptr + off + 1).to(tl.int32)
                B_val = lo | (hi << 8)
                tl.store(B_choices_out_ptr + i, B_val)
                off += 2
            else:
                err = tl.full((), 4, dtype=tl.int32)
                tl.store(B_choices_out_ptr + i, tl.full((), 0, dtype=tl.int32))
        else:
            tl.store(B_choices_out_ptr + i, tl.full((), 0, dtype=tl.int32))

        i += 1

    # header_bytes = 15 + 2 * num_B  (only meaningful if err == 0)
    if err == 0:
        header_bytes_i32 = 15 + (num_B_val * 2)

    # ---- store outputs ----
    tl.store(C_out_ptr, C_val)
    tl.store(K_out_ptr, K_val)
    tl.store(R_out_ptr, R_val)
    tl.store(num_B_out_ptr, num_B_val)
    tl.store(header_bytes_out_ptr, header_bytes_i32)
    tl.store(error_flag_ptr, err)


@triton.jit
def scan_rows_kernel(
    u8_payload_ptr,           # (total_bytes,) uint8
    row_bit_offsets_ptr,      # (num_rows,) int32  (bit offset of 16-bit length)
    row_payload_bytes_ptr,    # (num_rows,) int32
    best_B_idx_ptr,           # (num_rows,) int32
    use_bitmap_ptr,           # (num_rows,) int32 (0/1)
    header_end_bit: tl.int32,
    total_bits: tl.int32,
    num_rows: tl.int32,
    ROW_HEADER_BITS: tl.constexpr,
):
    """
    Sequential scan of all rows (1 program). For each row r:

      bit_off:  bit offset of 16-bit payload length
      length:   row_payload_bytes[r]
      header:   ((b_idx << 1) | use_bitmap) in ROW_HEADER_BITS bits
      rest:     payload_bits - ROW_HEADER_BITS bits

    Assumes the bitstream is valid and has enough bits for num_rows rows.
    """
    pid = tl.program_id(0)
    if pid != 0:
        return

    bit_off_i32 = header_end_bit
    r = tl.full((), 0, dtype=tl.int32)
    SIXTEEN_I32 = tl.full((), 16, dtype=tl.int32)

    while r < num_rows:
        # bit offset at the start of the 16-bit length for this row
        tl.store(row_bit_offsets_ptr + r, bit_off_i32)

        # read 16-bit payload length (bytes)
        length_u32, bit_off_after_len = read_nbits_triton(
            u8_payload_ptr, bit_off_i32, SIXTEEN_I32, total_bits
        )
        length_i32 = length_u32.to(tl.int32)
        tl.store(row_payload_bytes_ptr + r, length_i32)

        # read row header bits: ((b_idx << 1) | use_bitmap)
        header_u32, bit_off_after_header = read_nbits_triton(
            u8_payload_ptr,
            bit_off_after_len,
            tl.full((), ROW_HEADER_BITS, dtype=tl.int32),
            total_bits,
        )
        header_i32 = header_u32.to(tl.int32)
        use_bitmap_i32 = header_i32 & 1
        best_B_idx_i32 = header_i32 >> 1

        tl.store(best_B_idx_ptr + r, best_B_idx_i32)
        tl.store(use_bitmap_ptr + r, use_bitmap_i32)

        # skip remainder of this row's payload
        payload_bits_i32 = length_i32 * 8
        rem_bits_i32 = payload_bits_i32 - ROW_HEADER_BITS
        bit_off_i32 = bit_off_after_header + rem_bits_i32
        r += 1


@triton.jit
def decode_rows_kernel(
    u8_payload_ptr,           # (total_bytes,) uint8
    out_vals_ptr,             # (num_rows * K,) int32
    row_bit_offsets_ptr,      # (num_rows,) int32 (bit offset of 16-bit length)
    row_payload_bytes_ptr,    # (num_rows,) int32
    best_B_idx_ptr,           # (num_rows,) int32
    use_bitmap_ptr,           # (num_rows,) int32
    k_rice_choices_ptr,       # (num_B,) int32
    num_rows: tl.int32,
    K: tl.int32,
    ROW_HEADER_BITS: tl.constexpr,
):
    """
    Fully GPU decode of Rice/bitmap rows.

    For each row:
      - Start at bit offset of 16-bit length
      - Skip 16-bit length
      - Skip header bits (we already know b_idx/use_bitmap from scan)
      - First value: full Rice (unary + remainder)
      - Tail: Rice or bitmap+remainder
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    # Per-row metadata
    row_start_bit_i32 = tl.load(row_bit_offsets_ptr + row_idx).to(tl.int32)
    payload_bytes_i32 = tl.load(row_payload_bytes_ptr + row_idx).to(tl.int32)
    best_B_idx_i32 = tl.load(best_B_idx_ptr + row_idx).to(tl.int32)
    use_bitmap_i32 = (tl.load(use_bitmap_ptr + row_idx) & 1).to(tl.int32)

    # k_rice and M for this row
    k_rice_i32 = tl.load(k_rice_choices_ptr + best_B_idx_i32).to(tl.int32)
    M_i32 = (tl.full((), 1, dtype=tl.int32) << k_rice_i32)

    # Bit range of this row
    bit_after_len_i32 = row_start_bit_i32 + 16
    row_end_bit_i32 = bit_after_len_i32 + payload_bytes_i32 * 8

    # Skip header bits (we already know the contents)
    header_dummy_u32, bit_off_i32 = read_nbits_triton(
        u8_payload_ptr,
        bit_after_len_i32,
        tl.full((), ROW_HEADER_BITS, dtype=tl.int32),
        row_end_bit_i32,  # limit = end of this row
    )

    base_out = row_idx * K
    ONE_I32 = tl.full((), 1, dtype=tl.int32)

    # ---- first value: ALWAYS full Rice ----
    if K > 0:
        q0_i32, bit_off_i32, hit_end0_i32 = read_unary_bounded_triton(
            u8_payload_ptr,
            bit_off_i32,
            row_end_bit_i32,
        )
        r0_u32, bit_off_i32 = read_nbits_triton(
            u8_payload_ptr,
            bit_off_i32,
            k_rice_i32,
            row_end_bit_i32,  # limit
        )
        r0_i32 = r0_u32.to(tl.int32)
        v0_i32 = q0_i32 * M_i32 + r0_i32
        tl.store(out_vals_ptr + base_out, v0_i32)

    # ---- tail values ----
    i = tl.full((), 1, dtype=tl.int32)
    while i < K:
        if use_bitmap_i32 != 0:
            # Bitmap mode: q is 1 bit in {0,1}
            q_bit_u32, bit_off_i32 = read_nbits_triton(
                u8_payload_ptr,
                bit_off_i32,
                ONE_I32,
                row_end_bit_i32,
            )
            q_i32 = q_bit_u32.to(tl.int32)

            r_u32, bit_off_i32 = read_nbits_triton(
                u8_payload_ptr,
                bit_off_i32,
                k_rice_i32,
                row_end_bit_i32,
            )
            r_i32 = r_u32.to(tl.int32)
        else:
            # Full Rice mode
            q_i32, bit_off_i32, hit_end_i32 = read_unary_bounded_triton(
                u8_payload_ptr,
                bit_off_i32,
                row_end_bit_i32,
            )
            r_u32, bit_off_i32 = read_nbits_triton(
                u8_payload_ptr,
                bit_off_i32,
                k_rice_i32,
                row_end_bit_i32,
            )
            r_i32 = r_u32.to(tl.int32)

        v_i32 = q_i32 * M_i32 + r_i32
        tl.store(out_vals_ptr + base_out + i, v_i32)
        i += 1


def decode_batch_rows(
    payload: BytesLike,
    use_delta: bool = True,
    max_num_B: int = 16,
) -> tuple[torch.Tensor, int, int]:

    if not torch.cuda.is_available():
        raise RuntimeError("decode_batch_rows_gpu requires CUDA")

    # --- Move payload to CUDA (if needed) ---
    if isinstance(payload, torch.Tensor):
        assert payload.dtype == torch.uint8
        payload_gpu = payload if payload.is_cuda else payload.cuda()
    elif isinstance(payload, np.ndarray):
        assert payload.dtype == np.uint8
        payload_gpu = torch.from_numpy(payload).to("cuda", dtype=torch.uint8)
    elif isinstance(payload, (bytes, bytearray)):
        arr = np.frombuffer(bytes(payload), dtype=np.uint8)
        payload_gpu = torch.from_numpy(arr).to("cuda", dtype=torch.uint8)
    else:
        raise TypeError("Unsupported payload type")

    payload_gpu = payload_gpu.contiguous()
    dev = payload_gpu.device
    total_bytes = int(payload_gpu.numel())
    if total_bytes == 0:
        empty = torch.empty((0, 0), dtype=torch.int64, device=dev)
        return empty, 0, 0

    total_bits = total_bytes * 8

    # --- 1) Parse global header on GPU (now also gets num_rows = R) ---
    C_out = torch.empty(1, dtype=torch.int32, device=dev)
    K_out = torch.empty(1, dtype=torch.int32, device=dev)
    R_out = torch.empty(1, dtype=torch.int32, device=dev)      # NEW
    num_B_out = torch.empty(1, dtype=torch.int32, device=dev)
    B_choices_out = torch.empty(max_num_B, dtype=torch.int32, device=dev)
    header_bytes_out = torch.empty(1, dtype=torch.int32, device=dev)
    err_flag = torch.zeros(1, dtype=torch.int32, device=dev)

    parse_header_kernel[(1,)](
        payload_gpu,
        C_out,
        K_out,
        R_out,
        num_B_out,
        B_choices_out,
        header_bytes_out,
        err_flag,
        total_bytes,
        MAX_B_CHOICES=max_num_B,
    )

    torch.cuda.synchronize()
    err = int(err_flag.cpu().item())
    if err != 0:
        raise ValueError(f"parse_header_kernel failed with error code {err}")

    C = int(C_out.cpu().item())
    K = int(K_out.cpu().item())
    num_rows = int(R_out.cpu().item())           # NEW
    num_B = int(num_B_out.cpu().item())
    header_bytes = int(header_bytes_out.cpu().item())
    B_choices_list = [int(x) for x in B_choices_out[:num_B].cpu().tolist()]
    header_end_bit = header_bytes * 8

    # --- 2) Build k_rice choices on CPU -> move to GPU ---
    k_rice_choices = []
    for B in B_choices_list:
        M = C // B
        if M <= 0 or (M & (M - 1)) != 0:
            raise ValueError(f"M=C//B={M} not power of two for B={B}")
        k_rice_choices.append(int(math.log2(M)))
    k_rice_choices_tensor = torch.tensor(
        k_rice_choices, dtype=torch.int32, device=dev
    )

    B_choice_bits = (num_B - 1).bit_length()
    ROW_HEADER_BITS = 1 + B_choice_bits

    # --- 3) Scan rows on GPU to get per-row metadata ---
    row_bit_offsets = torch.empty(num_rows, dtype=torch.int32, device=dev)
    row_payload_bytes = torch.empty(num_rows, dtype=torch.int32, device=dev)
    best_B_idx = torch.empty(num_rows, dtype=torch.int32, device=dev)
    use_bitmap = torch.empty(num_rows, dtype=torch.int32, device=dev)

    scan_rows_kernel[(1,)](
        payload_gpu,
        row_bit_offsets,
        row_payload_bytes,
        best_B_idx,
        use_bitmap,
        header_end_bit,
        int(total_bits),
        int(num_rows),
        ROW_HEADER_BITS=ROW_HEADER_BITS,
    )

    # --- 4) Decode rows in parallel on GPU ---
    out_vals = torch.empty((num_rows, K), dtype=torch.int32, device=dev)
    decode_rows_kernel[(num_rows,)](
        payload_gpu,
        out_vals,
        row_bit_offsets,
        row_payload_bytes,
        best_B_idx,
        use_bitmap,
        k_rice_choices_tensor,
        int(num_rows),
        int(K),
        ROW_HEADER_BITS=ROW_HEADER_BITS,
    )

    # --- undo delta on-GPU if needed ---
    if use_delta:
        out_vals = torch.cumsum(out_vals, dim=1)
    return out_vals.to(torch.int64), C, num_rows


if __name__ == "__main__":
    torch.manual_seed(0)
    ROWS, K = 32, 16
    COLS = 4096

    x = torch.randn((ROWS, COLS), dtype=torch.float32, device="cuda")
    idx = torch.topk(x.abs(), k=K, dim=-1, largest=True, sorted=False).indices
    for use_delta in [False, True]:
        if use_delta:
            idx, _ = torch.sort(idx, dim=1)
        payload, _ = encode_batch_rows(idx, C=COLS, use_delta=use_delta, B_choices=(64, 128, 256))
        decoded, _, _ = decode_batch_rows(payload, use_delta=use_delta)
        dec = [torch.tensor(r, dtype=torch.int64) for r in decoded]
        ok = True
        for r in range(ROWS):
            if not torch.equal(torch.tensor(decoded[r]), idx[r]):
                ok = False
                print("Mismatch row", r)
                print("orig:", idx[r].tolist())
                print("dec :", decoded[r])
        print("Round-trip OK" if ok else "Round-trip MISMATCH")
