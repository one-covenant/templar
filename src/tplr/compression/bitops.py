from typing import Union

import numpy as np
import torch
import triton
import triton.language as tl

BytesLike = Union[bytes, bytearray, np.ndarray, torch.Tensor]

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
def write_nbits_fast(
    u8_ptr,
    bit_off_i32,   # start bit
    value_u32,     # LSB-first payload bits
    nbits_i32,     # 0..32
):
    # If nothing to write
    if nbits_i32 <= 0:
        return bit_off_i32

    start_bit  = bit_off_i32
    first_byte = (start_bit >> 3).to(tl.int32)
    first_bit  = (start_bit & 7).to(tl.int32)

    # How many bits fit in the first byte
    bits_in_first = tl.minimum(
        nbits_i32,
        tl.full((), 8, dtype=tl.int32) - first_bit,
    )

    # -------- leading partial byte --------
    if bits_in_first > 0:
        old_u8 = tl.load(u8_ptr + first_byte).to(tl.uint32)

        # mask for the bits we overwrite inside that byte
        mask_u32 = ((tl.full((), 1, tl.uint32) << bits_in_first) - 1) \
                   << first_bit

        # extract those bits from value_u32
        bits_u32 = (value_u32 & ((tl.full((), 1, tl.uint32) << bits_in_first) - 1)) \
                   << first_bit

        new_u8 = ((old_u8 & ~mask_u32) | bits_u32).to(tl.uint8)
        tl.store(u8_ptr + first_byte, new_u8)

        bit_off_i32 += bits_in_first
        value_u32 >>= bits_in_first
        nbits_i32  -= bits_in_first

    # Now bit_off_i32 is byte aligned (or nbits_i32 == 0)
    if nbits_i32 <= 0:
        return bit_off_i32

    cur_byte = (bit_off_i32 >> 3).to(tl.int32)

    # full bytes we can write
    full_bytes = (nbits_i32 >> 3).to(tl.int32)   # nbits_i32 // 8
    rem_bits   = (nbits_i32 & 7).to(tl.int32)

    # -------- full bytes --------
    jb = tl.full((), 0, dtype=tl.int32)
    while jb < full_bytes:
        # take lowest 8 bits from value_u32
        byte_val = (value_u32 & tl.full((), 0xFF, tl.uint32)).to(tl.uint8)
        tl.store(u8_ptr + cur_byte + jb, byte_val)
        value_u32 >>= 8
        jb += 1

    bit_off_i32 += full_bytes * 8

    # -------- trailing partial byte --------
    if rem_bits > 0:
        byte_idx = (bit_off_i32 >> 3).to(tl.int32)
        old_u8 = tl.load(u8_ptr + byte_idx).to(tl.uint32)

        mask_u32 = ( (tl.full((), 1, tl.uint32) << rem_bits) - 1 )
        bits_u32 = ( value_u32 & mask_u32 )

        new_u8 = ((old_u8 & ~mask_u32) | bits_u32).to(tl.uint8)
        tl.store(u8_ptr + byte_idx, new_u8)

        bit_off_i32 += rem_bits
    return bit_off_i32


@triton.jit
def read_nbits(u8_ptr, bit_off_i32, nbits_i32, limit_bit_i32):
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
def read_nbits_fast(u8_ptr, bit_off_i32, nbits_i32, limit_bit_i32):
    if nbits_i32 <= 0:
        return tl.full((), 0, tl.uint32), bit_off_i32

    # clamp to limit if you want to keep the defensive behavior
    max_bits = limit_bit_i32 - bit_off_i32
    nbits_i32 = tl.minimum(nbits_i32, max_bits)

    start_bit  = bit_off_i32
    end_bit    = bit_off_i32 + nbits_i32

    first_byte = (start_bit >> 3).to(tl.int32)
    first_bit  = (start_bit & 7).to(tl.int32)

    bits_in_first = tl.minimum(
        nbits_i32,
        tl.full((), 8, dtype=tl.int32) - first_bit,
    )

    val_u32 = tl.full((), 0, dtype=tl.uint32)
    shift   = tl.full((), 0, dtype=tl.int32)

    # -------- leading partial byte --------
    if bits_in_first > 0:
        byte = tl.load(u8_ptr + first_byte).to(tl.uint32)
        mask = ((tl.full((), 1, tl.uint32) << bits_in_first) - 1) << first_bit
        chunk = (byte & mask) >> first_bit
        val_u32 |= (chunk << shift)

        bit_off_i32 += bits_in_first
        shift       += bits_in_first
        nbits_i32   -= bits_in_first

    if nbits_i32 <= 0:
        return val_u32, bit_off_i32

    cur_byte = (bit_off_i32 >> 3).to(tl.int32)
    full_bytes = (nbits_i32 >> 3).to(tl.int32)
    rem_bits   = (nbits_i32 & 7).to(tl.int32)

    # -------- full bytes --------
    jb = tl.full((), 0, dtype=tl.int32)
    while jb < full_bytes:
        byte = tl.load(u8_ptr + cur_byte + jb).to(tl.uint32)
        val_u32 |= (byte << shift)
        shift   += 8
        jb      += 1

    bit_off_i32 += full_bytes * 8

    # -------- trailing partial byte --------
    if rem_bits > 0:
        byte = tl.load(u8_ptr + (bit_off_i32 >> 3).to(tl.int32)).to(tl.uint32)
        mask = (tl.full((), 1, tl.uint32) << rem_bits) - 1
        chunk = byte & mask
        val_u32 |= (chunk << shift)
        bit_off_i32 += rem_bits
    return val_u32, bit_off_i32


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