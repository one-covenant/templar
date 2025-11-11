# bits.py
# The MIT License (MIT)
# © 2025 tplr.ai
#
# Triton-powered Rice/bitmap encoder (per-row scheme) with a CPU decoder.
# Payload layout matches the CPU reference:
#   [ C-1 : 12b ][ N : 16b ][ reserved : 1b ]
#   then, for each row r=0..N-1:
#       [ row_len_bytes[r] : 16b ][ row_payload_bits[r] ]
#
# Dependencies: torch, triton (runtime), numpy (only for decode consumer code elsewhere if needed)

from __future__ import annotations
import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -------------------------- CPU decoder (unchanged format) --------------------------


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.idx = 0
        self.cur = 0
        self.nbits = 0

    def _fill(self, n: int) -> None:
        while self.nbits < n and self.idx < len(self.data):
            self.cur |= int(self.data[self.idx]) << self.nbits
            self.idx += 1
            self.nbits += 8

    def read_bits(self, n: int) -> int:
        if n <= 0:
            return 0
        self._fill(n)
        mask = (1 << n) - 1
        out = self.cur & mask
        self.cur >>= n
        self.nbits -= n
        return out

    def read_unary(self) -> int:
        q = 0
        while True:
            bit = self.read_bits(1)
            if bit == 0:
                break
            q += 1
        return q

    def read_bytes(self, n: int) -> bytes:
        out = bytearray()
        for _ in range(n):
            out.append(self.read_bits(8))
        return bytes(out)


def _rice_read(br: _BitReader, k: int) -> int:
    q = br.read_unary()
    r = br.read_bits(k) if k > 0 else 0
    return (q << k) + r


def decode_batch_rows(payload: bytes) -> tuple[list[list[int]], int, int]:
    """
    Decode payload created by encode_batch_rows(...).
    Returns (rows, C, N) where `rows` is a list of per-row global indices.
    """
    br = _BitReader(payload)
    C = br.read_bits(12) + 1
    N = br.read_bits(16)
    _ = br.read_bits(1)  # reserved

    rows: list[list[int]] = []
    for _i in range(N):
        row_len = br.read_bits(16)
        row_bytes = br.read_bytes(row_len)
        rr = _BitReader(row_bytes)
        lb = rr.read_bits(5)
        k_param = rr.read_bits(4)
        use_bitmap = rr.read_bits(1)
        B = 1 << lb
        n_sub = C // B

        indices: list[int] = []
        for j in range(n_sub):
            s_len = _rice_read(rr, k_param)
            if s_len == 0:
                continue
            if use_bitmap:
                bitmask = rr.read_bits(B)
                for loc in range(B):
                    if (bitmask >> loc) & 1:
                        indices.append(j * B + loc)
            else:
                for _ in range(s_len):
                    loc = rr.read_bits(lb)
                    indices.append(j * B + loc)
        rows.append(indices)
    return rows, C, N


# --------------------------- GPU-side param selection ---------------------------


def _rice_k_from_mean(lmbda: float) -> int:
    if lmbda <= 0.0:
        return 0
    return max(0, round(math.log2(max(lmbda, 1e-9))))


@torch.no_grad()
def _estimate_best_params_per_row(
    idx: torch.Tensor, C: int, B_choices: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch (GPU) estimate of best B, use_bitmap, and k_param per row.
    Mirrors your previous vectorised selector.
    """
    assert idx.dtype == torch.int64
    rows, k = idx.shape
    device = idx.device

    B_sorted = tuple(
        sorted([b for b in B_choices if b > 0 and (C % b) == 0 and (b & (b - 1)) == 0])
    )
    if not B_sorted:
        raise ValueError("No valid B choices for C")

    header = 5 + 4 + 1
    best_bits = torch.full((rows,), 1 << 60, device=device, dtype=torch.int64)
    best_B = torch.full((rows,), B_sorted[0], device=device, dtype=torch.int64)
    best_use_bitmap = torch.zeros((rows,), device=device, dtype=torch.bool)

    Bmin = B_sorted[0]
    if all((b % Bmin) == 0 for b in B_sorted):
        n_sub_min = C // Bmin
        js_min = (idx // Bmin).to(torch.int64)  # [rows, k]
        counts_min = (
            F.one_hot(js_min, num_classes=n_sub_min).sum(dim=1).to(torch.int64)
        )  # [rows, n_sub_min]
        lmbda_base = k / max(1, C)

        for B in B_sorted:
            g = B // Bmin
            counts_B = (
                counts_min
                if g == 1
                else counts_min.reshape(rows, n_sub_min // g, g).sum(dim=2)
            )
            lb = int(math.ceil(math.log2(B)))
            n_sub = C // B
            k_param = int(max(0, round(math.log2(max(lmbda_base * B, 1e-9)))))
            m = 1 << k_param
            q = counts_B // m
            rb_sum = q.sum(dim=1) + (1 + k_param) * n_sub
            nonzero = (counts_B > 0).sum(dim=1)
            bits_local = header + rb_sum + lb * k
            bits_bitmap = header + rb_sum + B * nonzero
            cur_bits = torch.minimum(bits_local, bits_bitmap).to(torch.int64)
            use_bitmap = bits_bitmap < bits_local
            update = cur_bits < best_bits
            best_bits = torch.where(update, cur_bits, best_bits)
            best_B = torch.where(update, torch.full_like(best_B, B), best_B)
            best_use_bitmap = torch.where(update, use_bitmap, best_use_bitmap)
    else:
        for B in B_sorted:
            lb = int(math.ceil(math.log2(B)))
            n_sub = C // B
            js = (idx // B).to(torch.int64)
            # Bincount per row with a constant bin width M=max_n_sub to avoid collisions
            M = C // min(B_sorted)
            row_ids = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
            flat = (row_ids * M + js).reshape(-1)
            counts = torch.bincount(flat, minlength=rows * M).reshape(rows, M)[
                :, :n_sub
            ]
            lmbda = (k / max(1, C)) * B
            k_param = int(max(0, round(math.log2(max(lmbda, 1e-9)))))
            m = 1 << k_param
            q = counts // m
            rb_sum = q.sum(dim=1) + (1 + k_param) * n_sub
            nonzero = (counts > 0).sum(dim=1)
            bits_local = header + rb_sum + lb * k
            bits_bitmap = header + rb_sum + B * nonzero
            cur_bits = torch.minimum(bits_local, bits_bitmap).to(torch.int64)
            use_bitmap = bits_bitmap < bits_local
            update = cur_bits < best_bits
            best_bits = torch.where(update, cur_bits, best_bits)
            best_B = torch.where(update, torch.full_like(best_B, B), best_B)
            best_use_bitmap = torch.where(update, use_bitmap, best_use_bitmap)

    # Rice k for chosen B
    lmbda = (idx.shape[1] / max(1, C)) * best_B.float()
    k_param = torch.clamp((lmbda.clamp_min(1e-9).log2().round()).to(torch.int64), min=0)

    return (
        best_B.to(torch.int32),
        best_use_bitmap.to(torch.uint8),
        k_param.to(torch.int32),
    )


# --------------------------- Triton helpers (no tl.uint32 calls) ---------------------------


@triton.jit
def _write_bits_u32(buf_ptr, bitpos, value, nbits):
    # Write 'nbits' LSBs from value, advance bitpos, and return the new bitpos.
    i = tl.zeros((), dtype=tl.int32)
    bp = bitpos.to(tl.int64)
    v = value.to(tl.int32)
    while i < nbits:
        bit = (v >> i) & 1
        byte_idx = bp // 8
        off = bp % 8
        p = buf_ptr + byte_idx
        old = tl.load(p, mask=True, other=0).to(tl.int32)
        newv = old | (bit << off)
        tl.store(p, newv.to(tl.uint8))
        bp += 1
        i += 1
    return bp


@triton.jit
def _write_unary(buf_ptr, bitpos, q):
    # Write q ones then a zero; zeros are already in buffer → just advance for trailing zero.
    bp = bitpos.to(tl.int64)
    i = tl.zeros((), dtype=tl.int32)
    while i < q:
        byte_idx = bp // 8
        off = bp % 8
        p = buf_ptr + byte_idx
        old = tl.load(p, mask=True, other=0).to(tl.int32)
        newv = old | (1 << off)
        tl.store(p, newv.to(tl.uint8))
        bp += 1
        i += 1
    # trailing zero bit: buffer is zero-initialized, so just skip one bit
    bp += 1
    return bp


@triton.jit
def _write_rice(buf_ptr, bitpos, x, kparam):
    # Golomb-Rice for non-negative x with parameter k.
    k = kparam.to(tl.int32)
    if k == 0:
        return _write_unary(buf_ptr, bitpos, x)
    m = 1 << k
    q = x // m
    r = x & (m - 1)
    bp = _write_unary(buf_ptr, bitpos, q)
    bp = _write_bits_u32(buf_ptr, bp, r, k)
    return bp


@triton.jit
def _set_one_bit(buf_ptr, bitpos):
    bp = bitpos.to(tl.int64)
    byte_idx = bp // 8
    off = bp % 8
    p = buf_ptr + byte_idx
    old = tl.load(p, mask=True, other=0).to(tl.int32)
    newv = old | (1 << off)
    tl.store(p, newv.to(tl.uint8))


# ------------------------------ Triton kernel ---------------------------------


@triton.jit
def _kernel_write_rows(
    idx_ptr,  # int64 [N*K]
    rows,
    k,
    C,  # ints
    bestB_ptr,  # int32 [N]
    usebm_ptr,  # uint8 [N] (0/1)
    kparam_ptr,  # int32 [N]
    row_bytes_ptr,  # int32 [N]
    len_bitpos_ptr,  # int64 [N]  (bitpos of 16-bit length)
    pay_bitpos_ptr,  # int64 [N]  (bitpos of first payload bit for the row)
    payload_ptr,  # uint8 [TOTAL_BYTES]
    K_MAX: tl.constexpr,  # upper bound for K (e.g., 256 or 512)
):
    r = tl.program_id(0)
    if r >= rows:
        return

    # Per-row params
    B = tl.load(bestB_ptr + r).to(tl.int32)
    use_bitmap = tl.load(usebm_ptr + r).to(tl.int1)
    kparam = tl.load(kparam_ptr + r).to(tl.int32)
    # B is power-of-two ⇒ lb = log2(B) exactly (cast to float for tl.log2)
    lb = tl.log2(B.to(tl.float32)).to(tl.int32)
    n_sub = C // B

    # Write 16-bit length at its position (interleaved layout)
    row_len = tl.load(row_bytes_ptr + r).to(tl.int32)
    len_bp = tl.load(len_bitpos_ptr + r)
    _ = _write_bits_u32(payload_ptr, len_bp, row_len, 16)

    # Row payload header
    bp = tl.load(pay_bitpos_ptr + r)
    bp = _write_bits_u32(payload_ptr, bp, lb, 5)  # lb
    bp = _write_bits_u32(payload_ptr, bp, kparam, 4)  # k
    bp = _write_bits_u32(payload_ptr, bp, use_bitmap.to(tl.int32), 1)  # mode

    # Emit each sub-chunk j in ascending order
    j = tl.zeros((), dtype=tl.int32)
    while j < n_sub:
        # -- first pass: count how many entries go to sub j
        got = tl.zeros((), dtype=tl.int32)
        t = tl.zeros((), dtype=tl.int32)
        while t < k:
            v = tl.load(idx_ptr + r * k + t).to(tl.int64)
            jj = (v // B).to(tl.int32)
            got += jj == j
            t += 1

        # write Rice length
        bp = _write_rice(payload_ptr, bp, got, kparam)

        if got > 0:
            if use_bitmap:
                # second pass: set bits for locations; advance by B bits
                start = bp
                t = tl.zeros((), dtype=tl.int32)
                while t < k:
                    v = tl.load(idx_ptr + r * k + t).to(tl.int64)
                    jj = (v // B).to(tl.int32)
                    if jj == j:
                        loc = (v - j.to(tl.int64) * B.to(tl.int64)).to(tl.int32)
                        _set_one_bit(payload_ptr, start + loc)
                    t += 1
                bp = start + B
            else:
                # local list: second pass writing lb-bit locs (order needn't be sorted)
                t = tl.zeros((), dtype=tl.int32)
                while t < k:
                    v = tl.load(idx_ptr + r * k + t).to(tl.int64)
                    jj = (v // B).to(tl.int32)
                    if jj == j:
                        loc = (v - j.to(tl.int64) * B.to(tl.int64)).to(tl.int32)
                        bp = _write_bits_u32(payload_ptr, bp, loc, lb)
                    t += 1
        j += 1
    # done


# -------------------------------- Public API ----------------------------------


@torch.no_grad()
def encode_batch_rows(
    idx: torch.Tensor,  # [rows, k] int64 (CUDA strongly recommended)
    *,
    C: int,
    B_choices: tuple[int, ...] = (64, 128),
) -> tuple[bytes, dict]:
    """
    Triton encoder for per-row Rice/bitmap codec.

    Returns:
      payload: bytes
      meta:    {total_bits, avg_bits_per_row, B_hist}
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. `pip install triton` and re-run.")

    if idx.dtype != torch.int64:
        idx = idx.to(torch.int64)
    if not idx.is_cuda:
        idx = idx.cuda()
    idx = idx.contiguous()

    rows, k = idx.shape
    device = idx.device

    # 1) Pick best params per row (GPU)
    best_B, use_bitmap, k_param = _estimate_best_params_per_row(
        idx, C=C, B_choices=B_choices
    )

    # 2) Compute exact per-row bit counts to size output
    header_bits_row = 5 + 4 + 1  # lb + k + mode
    lb = (best_B.float().log2().round().to(torch.int64)).clamp_min(
        1
    )  # exact for power-of-two
    n_sub = C // best_B.to(torch.int64)

    # Bincount per row with constant width M to prevent collisions
    M = int(max(C // int(b) for b in best_B.unique().tolist()))
    row_ids = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
    js = (idx // best_B.to(torch.int64).unsqueeze(1)).to(torch.int64)  # [rows, k]
    flat = (row_ids * M + js).reshape(-1)
    counts = torch.bincount(flat, minlength=rows * M).reshape(rows, M)  # [rows, M]
    # limit to effective n_sub per row when summing
    # rice bits: Σ(q + 1 + k) with q = c // 2^k
    m = 1 << k_param.to(torch.int64)
    q = counts // m.unsqueeze(1)
    rb_sum = q.sum(dim=1) + (1 + k_param.to(torch.int64)) * n_sub.to(torch.int64)
    nonzero = (counts > 0).sum(dim=1).to(torch.int64)

    bits_local = header_bits_row + rb_sum + lb * k
    bits_bitmap = header_bits_row + rb_sum + best_B.to(torch.int64) * nonzero
    row_bits = torch.minimum(bits_local, bits_bitmap).to(torch.int64)
    row_bytes = ((row_bits + 7) // 8).to(torch.int32)

    # 3) Allocate payload buffer (global header + interleaved [len16 | payload])
    total_bits_rows = int((16 * rows + 8 * row_bytes.sum().item()))
    total_bits = 12 + 16 + 1 + total_bits_rows
    total_bytes = (total_bits + 7) // 8
    payload = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

    # 4) Compute interleaved bit positions for each row
    header_bits = 12 + 16 + 1
    body_chunk_bits = 16 + 8 * row_bytes.to(torch.int64)  # [rows]
    prefix = torch.zeros_like(body_chunk_bits)
    if rows > 0:
        prefix[1:] = torch.cumsum(body_chunk_bits[:-1], dim=0)
    len_bitpos = header_bits + prefix  # [rows]
    pay_bitpos = len_bitpos + 16  # [rows]

    # 5) Write global header in-place (LSB-first) using torch ops
    def _write_scalar_bits(val: int, nbits: int, start_bit: int):
        v = int(val)
        bp = int(start_bit)
        nb = int(nbits)
        while nb > 0:
            byte_idx = bp // 8
            off = bp % 8
            take = min(nb, 8 - off)
            mask = ((v & ((1 << take) - 1)) << off) & 0xFF
            payload[byte_idx] |= torch.as_tensor(mask, dtype=torch.uint8, device=device)
            v >>= take
            bp += take
            nb -= take

    _write_scalar_bits(C - 1, 12, 0)
    _write_scalar_bits(rows, 16, 12)
    _write_scalar_bits(0, 1, 28)  # reserved

    # 6) Launch Triton to write all rows
    # Choose a safe K_MAX for your top-k; 256 covers k<=256; use 512 if you push k higher.
    K_MAX = 256 if k <= 256 else 512
    grid = (rows,)
    _kernel_write_rows[grid](
        idx,
        rows,
        k,
        C,
        best_B,
        use_bitmap,
        k_param,
        row_bytes,
        len_bitpos.to(torch.int64),
        pay_bitpos.to(torch.int64),
        payload,
        K_MAX=K_MAX,
    )

    # 7) Return bytes + meta
    payload_bytes = bytes(payload.detach().cpu().numpy().tobytes())
    B_hist = {int(b): int((best_B == b).sum().item()) for b in best_B.unique()}
    meta = {
        "total_bits": total_bits,
        "avg_bits_per_row": float(row_bits.float().mean().item()) if rows > 0 else 0.0,
        "B_hist": B_hist,
    }
    return payload_bytes, meta
