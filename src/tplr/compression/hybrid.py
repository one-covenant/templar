import math
import struct
from typing import Dict, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl


BytesLike = Union[bytes, bytearray, np.ndarray, torch.Tensor]


@torch.no_grad()
def encode_batch_rows(
        idx_sorted: torch.Tensor,
        *,
        C: int,
        B_choices: Tuple[int, ...] = (64, 128)
) -> Tuple[BytesLike, Dict]:
    """
    Compresses a 2D sorted int tensor of Top-K indices into a byte string
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

    if not isinstance(idx_sorted, torch.Tensor) or idx_sorted.ndim != 2:
        raise ValueError(f"idx must be a 2D int64 tensor, got {idx_sorted.shape} {idx_sorted.dtype}")

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

    # delta-encoding
    vals = torch.cat(
        (idx_sorted[:, :1], idx_sorted[:, 1:] - idx_sorted[:, :-1]),
        dim=1,
    ).to(torch.int32)

    # k_rice parameters (log2(C // B))
    k_rice_choices = tuple(int(math.log2(C // b)) for b in B_choices)
    num_B_choices = len(B_choices)
    k_rice_choices_tensor = torch.tensor(k_rice_choices, dtype=torch.int32, device=dev)

    # Row header bits
    B_choice_bits = (num_B_choices - 1).bit_length()
    ROW_HEADER_BITS = 1 + B_choice_bits

    # Output tensors for cost kernel
    costs = torch.empty((num_rows, num_B_choices), dtype=torch.int32, device=dev)
    is_bitmap = torch.empty((num_rows, num_B_choices), dtype=torch.int8, device=dev)

    # Calculate grid for cost kernel
    grid = (num_rows,)

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

    # Payload sizing
    row_payload_bits = min_costs
    row_payload_bytes = ((row_payload_bits + 7) // 8).to(torch.int32)

    if torch.any(row_payload_bytes > 0xFFFF):
        raise ValueError("Row payload length exceeds 65535 bytes; cannot store in uint16.")

    # Byte offsets
    if num_rows == 1:
        row_byte_offsets = torch.zeros(1, dtype=torch.int32, device=dev)
    else:
        row_byte_offsets = torch.nn.functional.pad(
            torch.cumsum(row_payload_bytes, dim=0, dtype=torch.int32)[:-1],
            (1, 0)
        )
    total_payload_bytes = int(row_payload_bytes.sum().item())

    # Global header Construction (CPU is faster for this small structured data)
    header_list = []
    header_list.append(b"CGRP")  # 4B magic
    header_list.append(struct.pack("<I", C))  # 4B C
    header_list.append(struct.pack("<H", k_dim))  # 2B K
    header_list.append(struct.pack("<I", num_rows))  # 4B R
    header_list.append(struct.pack("B", len(B_choices)))  # 1B num_B
    for b in B_choices:
        header_list.append(struct.pack("<H", b))  # 2B per B

    global_header_py = b"".join(header_list)
    global_header_len_bytes = len(global_header_py)

    # Row table layout
    row_entry_bytes = 3
    row_table_bytes = num_rows * row_entry_bytes
    payload_region_start = global_header_len_bytes + row_table_bytes
    final_buffer_bytes = payload_region_start + total_payload_bytes

    # Allocate full buffer
    payload_buf = torch.zeros(final_buffer_bytes, dtype=torch.uint8, device=dev)

    # Copy global header
    payload_buf[:global_header_len_bytes] = torch.tensor(
        list(global_header_py), dtype=torch.uint8, device=dev
    )

    # Build row table on GPU
    lengths_i32 = row_payload_bytes.to(torch.int32)
    headers_i32 = ((best_B_idx.to(torch.int32) << 1) | is_bitmap_choice).to(torch.int32)

    # Vectorized row table construction
    row_table_flat = torch.empty((num_rows, 3), dtype=torch.uint8, device=dev)
    row_table_flat[:, 0] = (lengths_i32 & 0xFF).to(torch.uint8)
    row_table_flat[:, 1] = ((lengths_i32 >> 8) & 0xFF).to(torch.uint8)
    row_table_flat[:, 2] = (headers_i32 & ((1 << ROW_HEADER_BITS) - 1)).to(torch.uint8)

    payload_buf[global_header_len_bytes: global_header_len_bytes + row_table_bytes] = row_table_flat.view(-1)

    # Calculate absolute byte offsets for pack kernel
    row_abs_byte_offsets = (payload_region_start + row_byte_offsets).to(torch.int32)

    # Pack payloads (Optimized Kernel)
    pack_kernel[(num_rows,)](
        vals,
        payload_buf,
        row_abs_byte_offsets,
        best_B_idx.to(torch.int32),
        is_bitmap_choice,
        k_rice_choices_tensor,
        num_rows,
        k_dim=k_dim,
    )

    # Meta stats
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
        delta_ptr,
        costs_ptr,
        is_bitmap_ptr,
        k_dim: tl.constexpr,
        num_rows: tl.int32,
        num_B_choices: tl.int32,
        k_rice_choices_ptr,
):
    """
    Calculates bit cost. One row per program instance.
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    i = tl.arange(0, k_dim)
    row_base = row_idx * k_dim
    delta = tl.load(delta_ptr + row_base + i)
    delta0 = tl.load(delta_ptr + row_base)

    b_idx = 0
    while b_idx < num_B_choices:
        k_rice = tl.load(k_rice_choices_ptr + b_idx)

        q = delta >> k_rice
        q0 = delta0 >> k_rice

        rice_cost = tl.sum(q + 1) + k_dim * k_rice

        # Bitmap cost: head is Rice, tail is (1 + k_rice)
        bitmap_cost = (q0 + 1 + k_rice) + (k_dim - 1) * (1 + k_rice)

        # Allow bitmap only if tail q are in {0,1}
        q_tail_max = tl.max(tl.where(i > 0, q, 0))
        bitmap_allowed = q_tail_max <= 1

        use_bitmap = (bitmap_cost < rice_cost) & bitmap_allowed
        min_cost = tl.where(use_bitmap, bitmap_cost, rice_cost)

        out_offset = row_idx * num_B_choices + b_idx
        tl.store(costs_ptr + out_offset, min_cost)
        tl.store(is_bitmap_ptr + out_offset, tl.where(use_bitmap, 1, 0).to(tl.int32))
        b_idx += 1


@triton.jit
def pack_kernel(
        delta_ptr,  # (rows, k_dim) IN int32
        u8_payload_ptr,  # OUT uint8
        row_abs_byte_offsets_ptr,  # (rows,) IN int32 (byte offset where payload starts)
        best_B_idx_ptr,  # (rows,) IN
        is_bitmap_ptr,  # (rows,) IN
        k_rice_choices_ptr,  # [num_B] IN
        num_rows: tl.int32,
        k_dim: tl.int32,  # dynamic
):
    """
    Writes payload bits using a 64-bit register accumulator.
    Modified to use unaligned byte stores to prevent cudaErrorMisalignedAddress.
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    # Load row params
    out_byte_off = tl.load(row_abs_byte_offsets_ptr + row_idx).to(tl.int32)
    b_idx_i32 = tl.load(best_B_idx_ptr + row_idx).to(tl.int32)
    use_bitmap_i32 = (tl.load(is_bitmap_ptr + row_idx) & 1).to(tl.int32)
    k_rice_i32 = tl.load(k_rice_choices_ptr + b_idx_i32).to(tl.int32)
    M_i32 = (tl.full((), 1, dtype=tl.int32) << k_rice_i32)

    # Accumulator state
    acc_data = tl.full((), 0, dtype=tl.uint64)
    acc_bits = tl.full((), 0, dtype=tl.int32)

    # Output pointer (byte-aligned)
    out_ptr_base = u8_payload_ptr + out_byte_off

    base_idx = row_idx * k_dim

    # ------------------------------------------------------------------
    # PROCESS LOOP
    # ------------------------------------------------------------------
    i = 0
    while i < k_dim:
        val = tl.load(delta_ptr + base_idx + i).to(tl.int32)

        # Compute q, r
        q = (val >> k_rice_i32).to(tl.uint64)
        r = (val & (M_i32 - 1)).to(tl.uint64)

        is_rice = (i == 0) | (use_bitmap_i32 == 0)

        if is_rice:
            # Rice: q '1's, then '0', then k_rice bits of r

            # Append Unary (q ones)
            q_count = q.to(tl.int32)
            while q_count > 0:
                acc_data |= (tl.full((), 1, dtype=tl.uint64) << acc_bits)
                acc_bits += 1
                q_count -= 1

                # Flush Check
                if acc_bits >= 32:
                    # Unaligned Store (4 bytes separately)
                    val_u32 = acc_data.to(tl.uint32)
                    tl.store(out_ptr_base + 0, (val_u32 & 0xFF).to(tl.uint8))
                    tl.store(out_ptr_base + 1, ((val_u32 >> 8) & 0xFF).to(tl.uint8))
                    tl.store(out_ptr_base + 2, ((val_u32 >> 16) & 0xFF).to(tl.uint8))
                    tl.store(out_ptr_base + 3, ((val_u32 >> 24) & 0xFF).to(tl.uint8))

                    out_ptr_base += 4
                    acc_data >>= 32
                    acc_bits -= 32

            # Append Separator '0'
            acc_bits += 1

        else:
            # Bitmap: q is 1 bit
            q_bit = tl.where(q > 0, 1, 0).to(tl.uint64)
            acc_data |= (q_bit << acc_bits)
            acc_bits += 1

        # Flush Check
        if acc_bits >= 32:
            val_u32 = acc_data.to(tl.uint32)
            tl.store(out_ptr_base + 0, (val_u32 & 0xFF).to(tl.uint8))
            tl.store(out_ptr_base + 1, ((val_u32 >> 8) & 0xFF).to(tl.uint8))
            tl.store(out_ptr_base + 2, ((val_u32 >> 16) & 0xFF).to(tl.uint8))
            tl.store(out_ptr_base + 3, ((val_u32 >> 24) & 0xFF).to(tl.uint8))

            out_ptr_base += 4
            acc_data >>= 32
            acc_bits -= 32

        # Append Remainder
        acc_data |= (r << acc_bits)
        acc_bits += k_rice_i32

        # Flush Check
        if acc_bits >= 32:
            val_u32 = acc_data.to(tl.uint32)
            tl.store(out_ptr_base + 0, (val_u32 & 0xFF).to(tl.uint8))
            tl.store(out_ptr_base + 1, ((val_u32 >> 8) & 0xFF).to(tl.uint8))
            tl.store(out_ptr_base + 2, ((val_u32 >> 16) & 0xFF).to(tl.uint8))
            tl.store(out_ptr_base + 3, ((val_u32 >> 24) & 0xFF).to(tl.uint8))

            out_ptr_base += 4
            acc_data >>= 32
            acc_bits -= 32

        i += 1

    # ------------------------------------------------------------------
    # FINAL FLUSH
    # ------------------------------------------------------------------
    # We might have 1..31 bits left. Write byte-by-byte.
    while acc_bits > 0:
        tl.store(out_ptr_base, (acc_data & 0xFF).to(tl.uint8))
        out_ptr_base += 1
        acc_data >>= 8
        acc_bits -= 8


@triton.jit
def parse_row_table_kernel(
        u8_payload_ptr,
        row_payload_bytes_ptr,
        best_B_idx_ptr,
        use_bitmap_ptr,
        row_table_start: tl.int32,
        num_rows: tl.int32,
        ROW_HEADER_BITS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_rows:
        return

    entry_offset = row_table_start + pid * 3

    b0 = tl.load(u8_payload_ptr + entry_offset).to(tl.int32)
    b1 = tl.load(u8_payload_ptr + entry_offset + 1).to(tl.int32)
    length_i32 = b0 | (b1 << 8)
    tl.store(row_payload_bytes_ptr + pid, length_i32)

    header_byte = tl.load(u8_payload_ptr + entry_offset + 2).to(tl.int32)
    header_mask = (tl.full((), 1, dtype=tl.int32) << ROW_HEADER_BITS) - 1
    header_i32 = header_byte & header_mask

    use_bitmap_i32 = header_i32 & 1
    best_B_idx_i32 = header_i32 >> 1

    tl.store(use_bitmap_ptr + pid, use_bitmap_i32)
    tl.store(best_B_idx_ptr + pid, best_B_idx_i32)


@triton.jit
def count_ones_in_word(word_u64):
    """
    Counts trailing ones in a 64-bit word (register level).
    Used for fast unary decoding without global memory access.
    """
    cnt = tl.full((), 0, dtype=tl.int32)
    ONE_U64 = tl.full((), 1, dtype=tl.uint64)

    check = word_u64
    cond = ((check & ONE_U64) == ONE_U64) & (cnt < 64)
    while cond:
        cnt += 1
        check >>= 1
        # Update condition for next iteration
        cond = ((check & ONE_U64) == ONE_U64) & (cnt < 64)
    return cnt


@triton.jit
def decode_rows_kernel(
        u8_payload_ptr,
        out_vals_ptr,
        row_bit_offsets_ptr,  # (rows,)
        row_payload_bytes_ptr,  # (rows,)
        best_B_idx_ptr,  # (rows,)
        use_bitmap_ptr,  # (rows,)
        k_rice_choices_ptr,  # (num_B,)
        num_rows: tl.int32,
        K: tl.int32,
):
    """
    Decodes rows using unaligned-safe 64-bit reads (via byte loads).
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    # Row params
    start_bit = tl.load(row_bit_offsets_ptr + row_idx).to(tl.int32)
    payload_bytes = tl.load(row_payload_bytes_ptr + row_idx).to(tl.int32)
    b_idx = tl.load(best_B_idx_ptr + row_idx).to(tl.int32)
    use_bitmap = (tl.load(use_bitmap_ptr + row_idx) & 1).to(tl.int32)

    k_rice = tl.load(k_rice_choices_ptr + b_idx).to(tl.int32)
    M = (tl.full((), 1, dtype=tl.int32) << k_rice)

    current_bit = start_bit
    base_out = row_idx * K

    i = 0
    while i < K:
        # ------------------------------------------------
        # BUFFERED LOAD (Unaligned Safe)
        # ------------------------------------------------
        byte_idx = current_bit // 8
        bit_in_byte = current_bit % 8

        # Manually load 8 bytes to form uint64.
        # This prevents misaligned access on all GPUs.
        b0 = tl.load(u8_payload_ptr + byte_idx + 0).to(tl.uint64)
        b1 = tl.load(u8_payload_ptr + byte_idx + 1).to(tl.uint64)
        b2 = tl.load(u8_payload_ptr + byte_idx + 2).to(tl.uint64)
        b3 = tl.load(u8_payload_ptr + byte_idx + 3).to(tl.uint64)
        b4 = tl.load(u8_payload_ptr + byte_idx + 4).to(tl.uint64)
        b5 = tl.load(u8_payload_ptr + byte_idx + 5).to(tl.uint64)
        b6 = tl.load(u8_payload_ptr + byte_idx + 6).to(tl.uint64)
        b7 = tl.load(u8_payload_ptr + byte_idx + 7).to(tl.uint64)

        word_u64 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | \
                   (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)

        # Shift out consumed bits
        stream = word_u64 >> bit_in_byte

        # ------------------------------------------------
        # DECODE LOGIC
        # ------------------------------------------------
        q = 0
        r = 0

        is_rice = (i == 0) | (use_bitmap == 0)

        bits_consumed = 0

        if is_rice:
            # Decode Unary q
            q = count_ones_in_word(stream)
            bits_consumed += (q + 1)
            stream >>= (q + 1)
        else:
            # Bitmap: q is single bit
            q = (stream & 1).to(tl.int32)
            bits_consumed += 1
            stream >>= 1

        # Decode Remainder r
        mask = (tl.full((), 1, dtype=tl.uint64) << k_rice) - 1
        r = (stream & mask).to(tl.int32)
        bits_consumed += k_rice

        # ------------------------------------------------
        # STORE
        # ------------------------------------------------
        val = q * M + r
        tl.store(out_vals_ptr + base_out + i, val)

        current_bit += bits_consumed
        i += 1


def decode_batch_rows(
        payload: BytesLike,
        max_num_B: int = 16,
) -> tuple[torch.Tensor, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    # Move to GPU/Tensor
    if isinstance(payload, torch.Tensor):
        payload_gpu = payload if payload.is_cuda else payload.cuda()
    elif isinstance(payload, np.ndarray):
        payload_gpu = torch.from_numpy(payload).to("cuda", dtype=torch.uint8)
    else:
        arr = np.frombuffer(bytes(payload), dtype=np.uint8)
        payload_gpu = torch.from_numpy(arr).to("cuda", dtype=torch.uint8)

    payload_gpu = payload_gpu.contiguous()
    dev = payload_gpu.device
    total_bytes = int(payload_gpu.numel())

    if total_bytes == 0:
        return torch.empty((0, 0), dtype=torch.int64, device=dev), 0, 0

    # --- 1) Parse Global Header (CPU) ---
    header_size_min = 15
    header_cpu = payload_gpu[:64].cpu().numpy().tobytes()

    try:
        # Fixed format string to match 15 bytes
        magic, C, K, num_rows, num_B = struct.unpack("<4sIHIB", header_cpu[:15])
    except struct.error:
        raise ValueError("Payload too short for header")

    if magic != b"CGRP":
        raise ValueError("Invalid magic bytes")

    offset = 15
    B_choices = []
    for _ in range(num_B):
        b_val = struct.unpack("<H", header_cpu[offset:offset + 2])[0]
        B_choices.append(b_val)
        offset += 2
    header_bytes = offset

    # --- 2) Prepare k_rice ---
    k_rice_choices = []
    for B in B_choices:
        M = C // B
        k_rice_choices.append(int(math.log2(M)))
    k_rice_choices_tensor = torch.tensor(k_rice_choices, dtype=torch.int32, device=dev)

    ROW_HEADER_BITS = 1 + (num_B - 1).bit_length()

    # --- 3) Parse Row Table (GPU) ---
    row_table_bytes = num_rows * 3

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

    # --- 4) Offsets ---
    payload_region_start = header_bytes + row_table_bytes

    row_payload_bytes_64 = row_payload_bytes.to(torch.int64)

    row_byte_offsets = torch.cumsum(row_payload_bytes_64, dim=0) - row_payload_bytes_64

    row_bit_offsets = (payload_region_start + row_byte_offsets).to(torch.int32) * 8

    # --- 5) Decode (GPU Optimized) ---
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