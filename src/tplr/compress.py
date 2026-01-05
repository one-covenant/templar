# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Adapted from https://github.com/bloc97/DeMo and NousResearch


# Global imports

import math
from typing import Generic, Literal, Sequence, TypeAlias, TypeVar, cast, overload

import torch
import torch.fft
from einops import rearrange
from torch.distributed.tensor import DTensor as DT

import tplr

# ─────────── type aliases ────────────────────────────────────────────────
# primitive shapes
ShapeT: TypeAlias = tuple[int, ...]  # original dense tensor shape
Shape4D = tuple[int, int, int, int]  # y, x, h, w
TotK: TypeAlias = int  # size of the last dim

# 12‑bit packed representation - just the uint8 buffer, no tuple
IdxT: TypeAlias = torch.Tensor  # 12-bit packed indices (stored as uint8 tensor)

QuantParamsT: TypeAlias = tuple[torch.Tensor, float, int, torch.Tensor, torch.dtype]

# For historical names kept elsewhere in the code
ValT: TypeAlias = torch.Tensor

# Boolean flag that propagates the chosen quantisation mode
Q = TypeVar("Q", Literal[True], Literal[False])


def pack_12bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack int64 indices into 12-bit representation.
    Every 2 indices (24 bits) are packed into 3 uint8 values.
    Assumes even number of indices (topk is always even).

    Args:
        indices: Tensor with values < 4096 (12-bit max), must have even number of elements

    Returns:
        packed_tensor as uint8
    """
    # Ensure indices fit in 12 bits
    max_idx = indices.max().item() if indices.numel() > 0 else 0
    if max_idx >= 4096:
        raise ValueError(f"Index {max_idx} exceeds 12-bit limit (4095)")

    # Flatten the tensor
    indices_flat = indices.flatten()
    n_indices = indices_flat.numel()

    # Ensure we have even number of indices
    if n_indices % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n_indices}")

    # Convert to int32 for bit manipulation
    indices_flat = indices_flat.to(torch.int32)

    # Process all as pairs
    indices_pairs = indices_flat
    n_pairs = n_indices // 2

    # Calculate packed size
    packed_size = n_pairs * 3
    packed = torch.zeros(packed_size, dtype=torch.uint8, device=indices.device)

    # Vectorized packing for pairs
    if n_pairs > 0:
        idx_pairs = indices_pairs.reshape(-1, 2)
        idx1 = idx_pairs[:, 0]
        idx2 = idx_pairs[:, 1]

        # Pack pairs: idx1 uses byte0 + lower 4 bits of byte1
        #            idx2 uses upper 4 bits of byte1 + byte2
        packed[0::3] = (idx1 & 0xFF).to(torch.uint8)  # Lower 8 bits of idx1
        packed[1::3] = (((idx1 >> 8) & 0x0F) | ((idx2 & 0x0F) << 4)).to(torch.uint8)
        packed[2::3] = ((idx2 >> 4) & 0xFF).to(torch.uint8)  # Upper 8 bits of idx2

    return packed


def unpack_12bit_indices(packed: torch.Tensor, values_shape: ShapeT) -> torch.Tensor:
    """
    Unpack 12-bit packed indices back to int64.
    Assumes even number of indices.

    Args:
        packed: Packed uint8 tensor
        values_shape: Shape of the values tensor (same as original indices shape)

    Returns:
        Unpacked indices as int64 tensor with original shape
    """
    n_indices = int(torch.prod(torch.tensor(values_shape)).item())

    if n_indices == 0:
        return torch.zeros(values_shape, dtype=torch.int64, device=packed.device)

    # Ensure even number of indices
    if n_indices % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n_indices}")

    # Prepare output
    indices = torch.zeros(n_indices, dtype=torch.int64, device=packed.device)

    # All indices are paired
    n_pairs = n_indices // 2

    if n_pairs > 0:
        # Vectorized unpacking
        byte0 = packed[0::3].to(torch.int64)
        byte1 = packed[1::3].to(torch.int64)
        byte2 = packed[2::3].to(torch.int64)

        # Reconstruct indices
        indices[0::2] = byte0 | ((byte1 & 0x0F) << 8)  # idx1
        indices[1::2] = ((byte1 >> 4) & 0x0F) | (byte2 << 4)  # idx2

    # Reshape to match values shape
    indices = indices.reshape(values_shape)

    return indices


class ChunkingTransformer:
    """
    A transformer for chunking tensors to enable more efficient gradient processing.

    This class handles the chunking of tensors into smaller blocks, which can be
    processed more efficiently. It pre-calculates Discrete Cosine Transform (DCT)
    basis matrices for various tensor sizes to speed up the transformation process.
    """

    @torch.no_grad()
    def __init__(self, model, target_chunk, norm="ortho"):
        """
        Initialise the ChunkingTransformer.

        Args:
            model: The model whose parameters will be processed.
            target_chunk (int): The target size for tensor chunks.
            norm (str): The normalization to be used for DCT ('ortho' or None).
        """
        self.target_chunk = target_chunk

        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()

        # Get all variants of model tensor sizes
        # Generate all possible valid DCT sizes for model tensors
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            for s in p.shape:
                # Get the closest smallest divisor to the targeted DCT size
                sc = _get_smaller_split(s, self.target_chunk)
                self.shape_dict[s] = sc

                # Pregenerate DCT basis matrices
                if sc not in self.f_dict:
                    I = torch.eye(sc)  # noqa: E741
                    self.f_dict[sc] = _dct(I, norm=norm).to(p.dtype).to(p.device)
                    self.b_dict[sc] = _idct(I, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None) -> torch.Tensor:
        """
        Apply a 2D einsum operation for encoding.

        Args:
            x (torch.Tensor): The input tensor.
            b (torch.Tensor): The first basis matrix.
            d (torch.Tensor, optional): The second basis matrix. Defaults to None.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ijbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None) -> torch.Tensor:
        """
        Apply a 2D einsum operation for decoding (transpose).

        Args:
            x (torch.Tensor): The input tensor.
            b (torch.Tensor): The first basis matrix.
            d (torch.Tensor, optional): The second basis matrix. Defaults to None.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijbd, bk, dl -> ...ijkl", x, b, d)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, use_dct: bool = False) -> torch.Tensor:
        """
        Encode a tensor by chunking and optionally applying DCT.

        Args:
            x (torch.Tensor): The input tensor to encode.
            use_dct (bool): Whether to apply the Discrete Cosine Transform.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y x h w", h=n1, w=n2)
            if use_dct:
                x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            if use_dct:
                x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor, *, use_dct: bool = False) -> torch.Tensor:
        """
        Decode a tensor by un-chunking and optionally applying inverse DCT.

        Args:
            x (torch.Tensor): The input tensor to decode.
            use_dct (bool): Whether to apply the inverse Discrete Cosine Transform.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        if len(x.shape) > 2:  # 2D weights
            if use_dct:
                n1 = x.shape[2]
                n2 = x.shape[3]
                n1w = self.b_dict[n1].to(x.device)
                n2w = self.b_dict[n2].to(x.device)
                self.b_dict[n1] = n1w
                self.b_dict[n2] = n2w

                x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y x h w -> (y h) (x w)")

        else:  # 1D weights
            if use_dct:
                n1 = x.shape[1]
                n1w = self.b_dict[n1].to(x.device)
                self.b_dict[n1] = n1w

                x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


class TopKCompressor(Generic[Q]):
    """
    A gradient sparsifier/compressor that uses Top-K selection and optional quantization.

    This class can be used to compress gradients by selecting the top-k largest values
    and optionally quantizing them to 8-bit integers for further size reduction.
    It supports both 1D and 2D tensors.
    """

    use_quantization: Q
    n_bins: int
    range_in_sigmas: int

    # ------------------------------------------------------------------ #
    # Constructor – two overloads so each instance "remembers" its mode
    # ------------------------------------------------------------------ #
    @overload
    def __init__(
        self: "TopKCompressor[Literal[True]]",
        *,
        use_quantization: Literal[True] = True,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None: ...

    @overload
    def __init__(
        self: "TopKCompressor[Literal[False]]",
        *,
        use_quantization: Literal[False] = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None: ...

    @torch.no_grad()
    def __init__(
        self,
        *,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None:
        """
        Initialise the TopKCompressor.

        Args:
            use_quantization (bool): Whether to use 8-bit quantization.
            quantization_bins (int): The number of bins for quantization.
            quantization_range (int): The quantization range in standard deviations.
        """
        self.use_quantization = cast(Q, use_quantization)
        if self.use_quantization:
            self.n_bins = quantization_bins
            self.range_in_sigmas = (
                quantization_range  # Quantization range in standard deviations
            )

    def _clamp_topk(self, x, topk) -> int:
        """
        Clamp the top-k value to be within the valid range and ensure it's even.

        Args:
            x (torch.Tensor): The input tensor.
            topk (int): The desired top-k value.

        Returns:
            int: The clamped and even top-k value.
        """
        topk = min(topk, x.shape[-1])
        topk = max(topk, 2)
        # Ensure topk is even for 12-bit packing efficiency
        topk = topk - (topk % 2)
        return int(topk)

    # ------------------------------------------------------------------ #
    # compress – returns a 5-tuple *or* a 4-tuple, depending on the mode
    # ------------------------------------------------------------------ #
    @overload
    def compress(
        self: "TopKCompressor[Literal[True]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK, QuantParamsT]: ...
    @overload
    def compress(
        self: "TopKCompressor[Literal[False]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK]: ...

    @torch.no_grad()
    def compress(self, x: torch.Tensor, topk: int):  # type: ignore[override]
        """
        Compress a tensor using top-k selection and optional quantization.

        Args:
            x (torch.Tensor): The input tensor to compress.
            topk (int): The number of top values to select.

        Returns:
            A tuple containing the compressed data. The format depends on whether
            quantization is used.
        """
        if isinstance(x, DT):  # check for dtensors
            x = x.to_local()
        xshape = x.shape

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        totalk = x.shape[-1]
        topk = self._clamp_topk(x, topk)

        idx_int64 = torch.topk(
            x.abs(), k=topk, dim=-1, largest=True, sorted=False
        ).indices
        val = torch.gather(x, dim=-1, index=idx_int64)

        # Pack indices into 12-bit representation for efficient storage
        # This reduces storage by 25% compared to int16
        idx = pack_12bit_indices(idx_int64)

        # Apply 8-bit quantization if enabled
        if self.use_quantization:
            val, quant_params = self._quantize_values(val)
            return idx, val, xshape, totalk, quant_params

        return idx, val, xshape, totalk

    @torch.no_grad()
    def decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor,
        val: torch.Tensor,
        xshape: ShapeT,
        totalk: int,
        quantize_params: QuantParamsT | None = None,
    ) -> torch.Tensor:
        """
        Decompress a tensor from its sparse representation.

        Args:
            p (torch.Tensor): A tensor with the target shape and device.
            idx (torch.Tensor): The indices of the non-zero values.
            val (torch.Tensor): The non-zero values.
            xshape (ShapeT): The original shape of the tensor.
            totalk (int): The total number of elements in the original tensor's last dim.
            quantize_params (QuantParamsT, optional): Quantization parameters. Defaults to None.

        Returns:
            torch.Tensor: The decompressed tensor.
        """
        if self.use_quantization and quantize_params is not None:
            val = self._dequantize_values(val, quantize_params)

        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Unpack 12-bit indices using val shape (if needed)
        if idx.dtype == torch.uint8:
            # 12-bit packed format - unpack it
            idx_int64 = unpack_12bit_indices(idx, val.shape)
        elif idx.dtype in (torch.int64, torch.long):
            # Already unpacked (from batch_decompress)
            idx_int64 = idx
        else:
            raise ValueError(
                f"Expected uint8 (packed) or int64 (unpacked) indices, got {idx.dtype}"
            )
        # Ensure val has the same dtype as x for scatter operation
        if val.dtype != x.dtype:
            val = val.to(dtype=x.dtype)

        x.scatter_reduce_(
            dim=-1, index=idx_int64, src=val, reduce="mean", include_self=False
        ).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            xshape4 = cast(Shape4D, xshape)
            h_dim = xshape4[2]
            x = rearrange(x, "y x (h w) -> y x h w", h=h_dim)

        return x

    @torch.no_grad()
    def batch_decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor | Sequence[torch.Tensor],
        val: torch.Tensor | Sequence[torch.Tensor],
        xshape: ShapeT,
        totalk: int,
        quantize_params: Sequence[QuantParamsT] | None = None,
        *,
        block_norms: torch.Tensor | None = None,
        normalise: bool = False,
        clip_norm: bool = True,
    ) -> torch.Tensor:
        """
        Decompress a batch of sparse tensors and combine them.

        Args:
            p (torch.Tensor): A tensor with the target shape and device.
            idx (torch.Tensor | Sequence[torch.Tensor]): A sequence of indices for each tensor in the batch.
            val (torch.Tensor | Sequence[torch.Tensor]): A sequence of values for each tensor in the batch.
            xshape (ShapeT): The original shape of the tensors.
            totalk (int): The total number of elements in the original tensor's last dim.
            quantize_params (Sequence[QuantParamsT], optional): A sequence of quantization parameters. Defaults to None.
            block_norms (torch.Tensor, optional): Pre-computed norms for each block. Defaults to None.
            normalise (bool): Whether to normalise the values. Defaults to False.
            clip_norm (bool): Whether to clip the norms of the values. Defaults to True.

        Returns:
            torch.Tensor: The combined, decompressed tensor.
        """
        if quantize_params is not None and not isinstance(quantize_params, list):
            quantize_params = [quantize_params] * len(val)  # type: ignore[list-item]

        processed_vals: list[torch.Tensor] = []
        dequant_vals = None
        norms = None
        clip_norm_val = None
        if self.use_quantization and quantize_params:
            dequant_vals = [
                self._dequantize_values(v, quantize_params[i])
                for i, v in enumerate(val)
            ]
        if clip_norm:
            # If caller already supplied per-block norms, trust them.
            if block_norms is not None:
                norms = block_norms.to(p.device)
            else:
                vals_for_norm = dequant_vals if dequant_vals is not None else val
                norms = torch.stack(
                    [torch.norm(sparse_vals, p=2) for sparse_vals in vals_for_norm]
                )
            clip_norm_val = torch.median(norms)

        vals = dequant_vals if dequant_vals is not None else val
        for i, v in enumerate(vals):
            v = v.to(p.device)

            if normalise:
                eps = 1e-8
                if len(v.shape) == 3:  # 2D weights
                    l2_norm = torch.norm(v, p=2, dim=2, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 2:  # 1D weights (biases)
                    l2_norm = torch.norm(v, p=2, dim=1, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 1:  # Single values
                    l2_norm = torch.norm(v, p=2)
                    if l2_norm > eps:
                        v = v / l2_norm
            elif clip_norm and norms is not None and clip_norm_val is not None:
                current_norm = norms[i]
                clip_factor = torch.clamp(clip_norm_val / (current_norm + 1e-8), max=1)
                v = v * clip_factor
            processed_vals.append(v)

        # Unpack and concatenate indices
        unpacked_indices = []
        val_list = val if isinstance(val, Sequence) else [val]
        idx_list = idx if isinstance(idx, Sequence) else [idx]

        for i, i_data in enumerate(idx_list):
            if i_data.dtype != torch.uint8:
                raise ValueError(
                    f"Expected uint8 for 12-bit packed indices, got {i_data.dtype}"
                )
            # Unpack 12-bit format using corresponding values shape
            v_data = val_list[i]
            idx_unpacked = unpack_12bit_indices(i_data.to(p.device), v_data.shape)
            unpacked_indices.append(idx_unpacked)

        idx_concat = torch.cat(unpacked_indices, dim=-1)
        val_concat = torch.cat(processed_vals, dim=-1).to(p.dtype)

        # Use decompress without quantization (since we already dequantized)
        return self.decompress(
            p, idx_concat, val_concat, xshape, totalk, quantize_params=None
        )

    @torch.no_grad()
    def _quantize_values(self, val: torch.Tensor) -> tuple[torch.Tensor, QuantParamsT]:
        """
        Quantize tensor values to 8-bit integers.

        Args:
            val (torch.Tensor): The tensor values to quantize.

        Returns:
            A tuple containing the quantized values (uint8) and the quantization parameters.
        """
        offset = self.n_bins // 2  # 128 for 8-bit
        shift = val.mean()
        centered = val - shift

        std = centered.norm() / math.sqrt(centered.numel() - 1)
        scale = self.range_in_sigmas * std / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered.dtype, device=val.device)

        centered_fp32 = centered.to(torch.float32)
        qval = (
            (centered_fp32 / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        device = qval.device
        sums = torch.zeros(self.n_bins, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_bins, dtype=torch.float32, device=device)

        sums.scatter_add_(0, qval.flatten().long(), centered_fp32.flatten())
        counts.scatter_add_(
            0, qval.flatten().long(), torch.ones_like(centered_fp32.flatten())
        )

        lookup = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        qparams: QuantParamsT = (shift, float(scale), offset, lookup, val.dtype)
        return qval, qparams

    @torch.no_grad()
    def _dequantize_values(
        self, val: torch.Tensor, qparams: QuantParamsT
    ) -> torch.Tensor:
        """
        Dequantize tensor values from 8-bit integers back to their original dtype.

        Args:
            val (torch.Tensor): The quantized values (uint8).
            qparams (QuantParamsT): The quantization parameters.

        Returns:
            torch.Tensor: The dequantized values.
        """
        if val.dtype == torch.uint8:
            shift, _, _, lookup, orig_dtype = qparams
            lookup = (
                lookup.to(val.device) if isinstance(lookup, torch.Tensor) else lookup
            )
            deq = lookup[val.long()] + shift
            val = deq.to(orig_dtype)
        return val

    def maybe_dequantize_values(
        self,
        vals: list[torch.Tensor],
        qparams: list[QuantParamsT],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """
        Dequantize a list of values if they are quantized.

        Args:
            vals (list[torch.Tensor]): A list of tensors that may be quantized.
            qparams (list[QuantParamsT]): A list of quantization parameters.
            device (torch.device): The device to move the tensors to.

        Returns:
            list[torch.Tensor]: A list of dequantized tensors.
        """
        if not isinstance(vals, (list, tuple)):
            vals = [vals]

        needs_dequantized = all([v.dtype == torch.uint8 for v in vals])
        if qparams is None or not needs_dequantized:
            return vals

        if (
            isinstance(qparams, tuple)
            and len(qparams) == 5  # potentially single or already 5 elements
            and not all([len(q) == 5 for q in qparams])  # already correctly formatted
        ):
            qparams = [qparams]

        if not isinstance(qparams, list):
            qparams = [qparams]

        vals_f32: list[torch.Tensor] = []
        for i, v in enumerate(vals):
            v = v.to(device)
            if v.dtype == torch.uint8:  # still quantised → decode
                if qparams[i] is None:
                    tplr.logger.warning(f"Missing quant_params for vals[{i}]]; skip.")
                    break
                qp = qparams[i]
                v = self._dequantize_values(v, qp).to(device)
            vals_f32.append(v)

        if len(vals_f32) != len(vals):  # some decode failed
            raise IndexError(
                f"Mismatch in val lengths: dequant({len(vals_f32)}) vs original({len(vals)})"
            )

        return vals_f32


# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v) -> torch.Tensor:
    """FFT-based implementation of the DCT."""
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V) -> torch.Tensor:
    """IRFFT-based implementation of the IDCT."""
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None) -> torch.Tensor:
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None) -> torch.Tensor:
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * math.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def _get_prime_divisors(n: int) -> list[int]:
    """
    Get the prime divisors of a number.

    Args:
        n (int): The number to factorize.

    Returns:
        list[int]: A list of prime divisors.
    """
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n: int) -> list[int]:
    """
    Get all divisors of a number.

    Args:
        n (int): The number to get divisors for.

    Returns:
        list[int]: A sorted list of divisors.
    """
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        # Find all the products that are divisors of n
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n: int, close_to: int) -> int:
    """
    Find the largest divisor of n that is less than or equal to close_to.

    Args:
        n (int): The number to find a divisor for.
        close_to (int): The target value to be close to.

    Returns:
        int: The largest divisor of n that is <= close_to.
    """
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n
