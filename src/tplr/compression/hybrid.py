import math
from typing import Dict
from typing import Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from .bitops import write_nbits_fast, read_unary_bounded_triton, read_nbits_fast

BytesLike = Union[bytes, bytearray, np.ndarray, torch.Tensor]


@torch.no_grad()
def encode_batch_rows(
        idx: torch.Tensor,
        *,
        C: int,
        B_choices: Tuple[int, ...] = (64, 128)
) -> Tuple[BytesLike, Dict]:
    """
    Compresses a 2D int64 tensor of Top-K indices into a byte string
    using a per-row adaptive Rice/Bitmap compression scheme on the GPU.

    Layout:

    [global header]
      0..3   : "CGRP"         (magic)
      4..7   : C      (uint32 LE)
      8..9   : K      (uint16 LE)
      10..13 : R      (uint32 LE, num_rows)
      14     : num_B  (uint8)
      15..   : B_choices (num_B * uint16 LE)

    [row table] (num_rows entries, 3 bytes each)
      - uint16 length_bytes[r]  (payload size in BYTES for row r)
      - uint8  header[r]        ((best_B_idx << 1) | use_bitmap)

    [payload region]
      - concatenated bitstreams, one per row, each length_bytes[r] bytes,
        byte-aligned, containing ONLY the Rice/bitmap-coded deltas
        (no per-row length or header in-band).
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

    vals = torch.cat(
        (idx[:, :1], idx[:, 1:] - idx[:, :-1]),
        dim=1,
    )

    # Cast to int32 for Triton kernels
    vals = vals.to(torch.int32)

    # k_rice parameters (log2(C // B))
    k_rice_choices = tuple(int(math.log2(C // b)) for b in B_choices)
    num_B_choices = len(B_choices)
    k_rice_choices_tensor = torch.tensor(k_rice_choices, dtype=torch.int32, device=dev)

    # Row header bits (only used for packing row-table header byte)
    B_choice_bits = (num_B_choices - 1).bit_length()
    ROW_HEADER_BITS = 1 + B_choice_bits  # (best_B_idx << 1) | use_bitmap

    # Output tensors for cost kernel
    costs = torch.empty((num_rows, num_B_choices), dtype=torch.int32, device=dev)
    is_bitmap = torch.empty((num_rows, num_B_choices), dtype=torch.int8, device=dev)
    grid = (num_rows,)

    # cost kernel: bits required for deltas only (no header bits)
    cost_kernel[grid](
        vals,
        costs,
        is_bitmap,
        k_dim=k_dim,
        num_rows=num_rows,
        num_B_choices=num_B_choices,
        k_rice_choices_ptr=k_rice_choices_tensor,
    )

    # Best choice per row
    min_costs, best_B_idx = torch.min(costs, dim=1)
    is_bitmap_choice = torch.gather(is_bitmap, 1, best_B_idx.unsqueeze(1)).squeeze(1).to(torch.int32)

    # (1) payload bits per row = bits for deltas only
    row_payload_bits = min_costs  # (rows,)

    # (2) payload bytes per row (rounded up)
    row_payload_bytes = ((row_payload_bits + 7) // 8).to(torch.int32)  # (rows,)

    # ensure fit in uint16 for the row table
    if torch.any(row_payload_bytes > 0xFFFF):
        raise ValueError("Row payload length exceeds 65535 bytes; cannot store in uint16.")

    # byte offsets within the payload region (no gaps)
    # row_byte_offsets[r] = sum_{i<r} row_payload_bytes[i]
    if num_rows == 1:
        row_byte_offsets = torch.zeros(1, dtype=torch.int32, device=dev)
    else:
        row_byte_offsets = torch.nn.functional.pad(
            torch.cumsum(row_payload_bytes, dim=0, dtype=torch.int32)[:-1],
            (1, 0)
        )
    total_payload_bytes = int(row_payload_bytes.sum().item())

    # Global header bytes
    header_list = []
    header_list.append(b"CGRP")  # 4B magic
    header_list.append(int(C).to_bytes(4, "little"))         # 4B C (uint32 LE)
    header_list.append(int(k_dim).to_bytes(2, "little"))     # 2B K (uint16 LE)
    header_list.append(int(num_rows).to_bytes(4, "little"))  # 4B R (uint32 LE)
    header_list.append(bytes([len(B_choices)]))              # 1B num_B
    for b in B_choices:
        header_list.append(int(b).to_bytes(2, "little"))     # 2B per B (uint16 LE)

    global_header_py = b"".join(header_list)
    global_header_len_bytes = len(global_header_py)  # 15 + 2 * len(B_choices)

    # row table layout: 3 bytes per row
    row_entry_bytes = 3
    row_table_bytes = num_rows * row_entry_bytes
    payload_region_start = global_header_len_bytes + row_table_bytes
    final_buffer_bytes = payload_region_start + total_payload_bytes

    # Allocate full buffer
    payload_buf = torch.zeros(final_buffer_bytes, dtype=torch.uint8, device=dev)

    # Write global header
    payload_buf[:global_header_len_bytes] = torch.tensor(
        list(global_header_py), dtype=torch.uint8, device=dev
    )

    # build row table on GPU
    # length_bytes (uint16 LE) + header byte
    lengths_i32 = row_payload_bytes.to(torch.int32)
    headers_i32 = ((best_B_idx.to(torch.int32) << 1) | is_bitmap_choice).to(torch.int32)

    row_table = torch.empty((num_rows, row_entry_bytes), dtype=torch.uint8, device=dev)
    row_table[:, 0] = (lengths_i32 & 0xFF).to(torch.uint8)
    row_table[:, 1] = ((lengths_i32 >> 8) & 0xFF).to(torch.uint8)

    # Only the low ROW_HEADER_BITS bits are meaningful, but we just store the byte.
    row_table[:, 2] = (headers_i32 & ((1 << ROW_HEADER_BITS) - 1)).to(torch.uint8)

    payload_buf[
        global_header_len_bytes : global_header_len_bytes + row_table_bytes
    ] = row_table.view(-1)

    # compute bit offsets for each row's payload (no per-row length/header in-band)
    row_bit_offsets = (payload_region_start + row_byte_offsets).to(torch.int32) * 8

    # pack payloads
    pack_kernel[(num_rows,)](
        vals,
        payload_buf,
        row_bit_offsets,
        best_B_idx.to(torch.int32),
        is_bitmap_choice,  # int32 0/1
        k_rice_choices_tensor,
        num_rows,
        k_dim=k_dim,
    )

    # meta
    b_counts = torch.bincount(best_B_idx, minlength=len(B_choices))
    B_hist = {b: c.item() for b, c in zip(B_choices, b_counts)}
    total_row_bytes = total_payload_bytes + row_entry_bytes * num_rows
    total_bits = int(total_row_bytes * 8)

    meta = {
        "total_bits": total_bits,
        "avg_bits_per_row": float(total_bits / num_rows),
        "avg_payload_bits_per_row": float(row_payload_bits.float().mean().item()),
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
def pack_kernel(
    delta_ptr,              # (rows, k_dim) IN int32
    u8_payload_ptr,         # (final_buffer_bytes,) OUT uint8
    row_bit_offsets_ptr,    # (rows,) IN  int32 (bit offset where payload starts)
    best_B_idx_ptr,         # (rows,) IN  int32
    is_bitmap_ptr,          # (rows,) IN  int32 (0/1)
    k_rice_choices_ptr,     # [num_B] IN int32
    num_rows: tl.int32,
    k_dim: tl.int32,        # dynamic
):
    """
    Writes only the Rice/bitmap-coded payload bits for each row.

    Each program instance handles one row. Bit order is LSB-first.
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    # Per-row meta
    bit_off_i32 = tl.load(row_bit_offsets_ptr + row_idx).to(tl.int32)
    b_idx_i32 = tl.load(best_B_idx_ptr + row_idx).to(tl.int32)
    use_bitmap_i32 = (tl.load(is_bitmap_ptr + row_idx) & 1).to(tl.int32)

    # params
    k_rice_i32 = tl.load(k_rice_choices_ptr + b_idx_i32).to(tl.int32)
    M_i32 = (tl.full((), 1, dtype=tl.int32) << k_rice_i32)

    ONE_U32 = tl.full((), 1, dtype=tl.uint32)
    ZERO_U32 = tl.full((), 0, dtype=tl.uint32)
    ONE_I32 = tl.full((), 1, dtype=tl.int32)
    THIRTY_ONE_I32 = tl.full((), 31, dtype=tl.int32)

    base = row_idx * k_dim

    # ---- first delta: ALWAYS full Rice (unary + remainder) ----
    if k_dim > 0:
        v0 = tl.load(delta_ptr + base).to(tl.int32)
        q0 = (v0 >> k_rice_i32).to(tl.int32)
        r0 = (v0 & (M_i32 - 1)).to(tl.int32)

        # q0 ones in chunks of <= 31, then a single 0
        q_left = q0
        while q_left > 0:
            chunk = tl.minimum(q_left, THIRTY_ONE_I32)
            ones = (ONE_U32 << chunk) - ONE_U32
            bit_off_i32 = write_nbits_fast(u8_payload_ptr, bit_off_i32, ones, chunk)
            q_left -= chunk

        # terminating 0 bit
        bit_off_i32 = write_nbits_fast(u8_payload_ptr, bit_off_i32, ZERO_U32, ONE_I32)
        # remainder
        bit_off_i32 = write_nbits_fast(
            u8_payload_ptr, bit_off_i32, r0.to(tl.uint32), k_rice_i32
        )

    # ---- tail deltas ----
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
            bit_off_i32 = write_nbits_fast(u8_payload_ptr, bit_off_i32, ones, chunk)
            q_left -= chunk

        # terminating 0 bit only in full-Rice mode
        n_term = tl.where(use_bitmap_i32 != 0, tl.full((), 0, dtype=tl.int32), ONE_I32)
        bit_off_i32 = write_nbits_fast(u8_payload_ptr, bit_off_i32, ZERO_U32, n_term)

        # bitmap q only if bitmap
        q_bit = tl.where(q > 0, ONE_U32, ZERO_U32)
        n_qbit = tl.where(use_bitmap_i32 != 0, ONE_I32, tl.full((), 0, dtype=tl.int32))
        bit_off_i32 = write_nbits_fast(u8_payload_ptr, bit_off_i32, q_bit, n_qbit)

        # remainder always
        bit_off_i32 = write_nbits_fast(u8_payload_ptr, bit_off_i32, r.to(tl.uint32), k_rice_i32)
        i += 1


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
def parse_row_table_kernel(
    u8_payload_ptr,           # (total_bytes,) uint8
    row_payload_bytes_ptr,    # (num_rows,) int32
    best_B_idx_ptr,           # (num_rows,) int32
    use_bitmap_ptr,           # (num_rows,) int32
    row_table_start: tl.int32,
    num_rows: tl.int32,
    ROW_HEADER_BITS: tl.constexpr,
):
    """
    Parse the row table:

      For each row r:
        offset = row_table_start + r * 3
        length_bytes[r] = uint16 LE at offset
        header_byte     = uint8 at offset + 2
        header_bits     = header_byte & ((1 << ROW_HEADER_BITS) - 1)
        use_bitmap[r]   = header_bits & 1
        best_B_idx[r]   = header_bits >> 1
    """
    pid = tl.program_id(0)
    if pid >= num_rows:
        return

    entry_offset = row_table_start + pid * 3

    # length_bytes: uint16 LE
    b0 = tl.load(u8_payload_ptr + entry_offset).to(tl.int32)
    b1 = tl.load(u8_payload_ptr + entry_offset + 1).to(tl.int32)
    length_i32 = b0 | (b1 << 8)
    tl.store(row_payload_bytes_ptr + pid, length_i32)

    # header byte
    header_byte = tl.load(u8_payload_ptr + entry_offset + 2).to(tl.int32)
    header_mask = (tl.full((), 1, dtype=tl.int32) << ROW_HEADER_BITS) - 1
    header_i32 = header_byte & header_mask

    use_bitmap_i32 = header_i32 & 1
    best_B_idx_i32 = header_i32 >> 1

    tl.store(use_bitmap_ptr + pid, use_bitmap_i32)
    tl.store(best_B_idx_ptr + pid, best_B_idx_i32)



@triton.jit
def decode_rows_kernel(
    u8_payload_ptr,           # (total_bytes,) uint8
    out_vals_ptr,             # (num_rows * K,) int32
    row_bit_offsets_ptr,      # (num_rows,) int32 (bit offset of first encoded bit)
    row_payload_bytes_ptr,    # (num_rows,) int32
    best_B_idx_ptr,           # (num_rows,) int32
    use_bitmap_ptr,           # (num_rows,) int32
    k_rice_choices_ptr,       # (num_B,) int32
    num_rows: tl.int32,
    K: tl.int32,
):
    """
    Fully GPU decode of Rice/bitmap rows.

    For each row r:
      - Bit range:
          start_bit = row_bit_offsets[r]
          end_bit   = start_bit + row_payload_bytes[r] * 8
      - First value: full Rice (unary + remainder)
      - Tail: Rice or bitmap+remainder depending on use_bitmap[r].
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
    row_end_bit_i32 = row_start_bit_i32 + payload_bytes_i32 * 8

    base_out = row_idx * K
    ONE_I32 = tl.full((), 1, dtype=tl.int32)

    bit_off_i32 = row_start_bit_i32

    # ---- first value: ALWAYS full Rice ----
    if K > 0:
        q0_i32, bit_off_i32, hit_end0_i32 = read_unary_bounded_triton(
            u8_payload_ptr,
            bit_off_i32,
            row_end_bit_i32,
        )
        r0_u32, bit_off_i32 = read_nbits_fast(
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
            q_bit_u32, bit_off_i32 = read_nbits_fast(
                u8_payload_ptr,
                bit_off_i32,
                ONE_I32,
                row_end_bit_i32,
            )
            q_i32 = q_bit_u32.to(tl.int32)

            r_u32, bit_off_i32 = read_nbits_fast(
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
            r_u32, bit_off_i32 = read_nbits_fast(
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

    # --- 1) Parse global header on GPU ---
    C_out = torch.empty(1, dtype=torch.int32, device=dev)
    K_out = torch.empty(1, dtype=torch.int32, device=dev)
    R_out = torch.empty(1, dtype=torch.int32, device=dev)
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
    num_rows = int(R_out.cpu().item())
    num_B = int(num_B_out.cpu().item())
    header_bytes = int(header_bytes_out.cpu().item())
    B_choices_list = [int(x) for x in B_choices_out[:num_B].cpu().tolist()]

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

    # --- 3) Parse row table on GPU ---
    row_entry_bytes = 3
    row_table_bytes = num_rows * row_entry_bytes
    if header_bytes + row_table_bytes > total_bytes:
        raise ValueError("Truncated payload: row table exceeds payload length")

    row_payload_bytes = torch.empty(num_rows, dtype=torch.int32, device=dev)
    best_B_idx = torch.empty(num_rows, dtype=torch.int32, device=dev)
    use_bitmap = torch.empty(num_rows, dtype=torch.int32, device=dev)

    parse_row_table_kernel[(num_rows,)](
        payload_gpu,
        row_payload_bytes,
        best_B_idx,
        use_bitmap,
        int(header_bytes),
        int(num_rows),
        ROW_HEADER_BITS=ROW_HEADER_BITS,
    )

    # --- 4) Compute per-row bit offsets into payload region ---
    payload_region_start_byte = header_bytes + row_table_bytes
    if payload_region_start_byte > total_bytes:
        raise ValueError("Truncated payload: missing payload region")

    # byte offsets within the payload region
    row_payload_bytes_64 = row_payload_bytes.to(torch.int64)
    row_byte_offsets = torch.cumsum(row_payload_bytes_64, dim=0) - row_payload_bytes_64

    # Sanity check: last row must end within the buffer
    last_end = int(
        payload_region_start_byte
        + row_byte_offsets[-1].item()
        + row_payload_bytes_64[-1].item()
    )
    if last_end > total_bytes:
        raise ValueError("Truncated payload: row payload bytes exceed buffer length")

    row_bit_offsets = (
        payload_region_start_byte + row_byte_offsets
    ).to(torch.int32) * 8

    # --- 5) Decode rows in parallel on GPU ---
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
    )

    out_vals = torch.cumsum(out_vals, dim=1)
    return out_vals.to(torch.int64), C, num_rows

