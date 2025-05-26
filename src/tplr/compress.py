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

import torch
import torch.fft
from einops import rearrange


class TransformDCT:
    @torch.no_grad()
    def __init__(self, model, target_chunk, norm="ortho"):
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
    def einsum_2d(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ibjd", x, b, d)

    @torch.no_grad()
    def encode(self, x):
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x):
        if len(x.shape) > 2:  # 2D weights
            n1 = x.shape[2]
            n2 = x.shape[3]
            n1w = self.b_dict[n1].to(x.device)
            n2w = self.b_dict[n2].to(x.device)
            self.b_dict[n1] = n1w
            self.b_dict[n2] = n2w

            x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y h x w -> (y h) (x w)")

        else:  # 1D weights
            n1 = x.shape[1]
            n1w = self.b_dict[n1].to(x.device)
            self.b_dict[n1] = n1w

            x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


class CompressDCT:
    @torch.no_grad()
    def __init__(
        self,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ):
        self.use_quantization = use_quantization
        if self.use_quantization:
            self.n_bins = quantization_bins
            self.range_in_sigmas = (
                quantization_range  # Quantization range in standard deviations
            )

    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return topk

    @torch.no_grad()
    def compress(self, x, topk):
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

        # Cast idx to int16 for saving or transmission
        idx = idx_int64.to(torch.int16)

        # Apply 8-bit quantization if enabled
        if self.use_quantization:
            val, quant_params = self._quantize_values(val)
            return idx, val, xshape, totalk, quant_params

        return idx, val, xshape, totalk, None

    @torch.no_grad()
    def decompress(self, p, idx, val, xshape, totalk, quantize_params=None):
        # Dequantize if values were quantized
        if self.use_quantization and quantize_params is not None:
            val = self._dequantize_values(val, quantize_params)

        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Cast back to int64 before using scatter/gather
        idx_int64 = idx.to(torch.int64)
        x.scatter_reduce_(
            dim=-1, index=idx_int64, src=val, reduce="mean", include_self=False
        ).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x (h w) -> y x h w", h=xshape[2])

        return x

    @torch.no_grad()
    def batch_decompress(
        self, p, idx, val, xshape, totalk, quantize_params=None, normalise=True
    ):
        """
        Decompress multiple tensors in batch mode.
        """
        # Ensure idx and val are lists
        if not isinstance(idx, list):
            idx = [idx]
        if not isinstance(val, list):
            val = [val]

        # Handle quantization parameters
        if quantize_params is not None:
            if not isinstance(quantize_params, list):
                quantize_params = [quantize_params] * len(val)

        # Process values - dequantize if needed
        processed_vals = []
        for i in range(len(val)):
            v = val[i].to(p.device)

            # Dequantize if we have quantization parameters
            if self.use_quantization and quantize_params and i < len(quantize_params):
                v = self._dequantize_values(v, quantize_params[i])

            # Apply L2 normalization to this individual tensor's values
            # Normalize along the last dimension (where top-k was selected)
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

            processed_vals.append(v)

        # Concatenate everything
        idx_concat = torch.cat([i.to(p.device) for i in idx], dim=-1)
        val_concat = torch.cat(processed_vals, dim=-1).to(p.dtype)

        # Use decompress without quantization (since we already dequantized)
        return self.decompress(
            p, idx_concat, val_concat, xshape, totalk, quantize_params=None
        )

    @torch.no_grad()
    def _quantize_values(self, val):
        """
        Quantize values to 8-bit representation with statistical approach

        Args:
            val: Tensor of values to quantize

        Returns:
            tuple: (quantized_values, quantization_parameters)
        """
        # Statistical quantization approach
        offset = self.n_bins // 2  # 128 for 8-bit
        shift = val.mean()

        # Center tensor around mean
        centered_val = val - shift

        # Calculate standard deviation (unbiased)
        std_unbiased = centered_val.norm() / math.sqrt(centered_val.numel() - 1)

        # Compute scale factor based on standard deviation range
        scale = self.range_in_sigmas * std_unbiased / self.n_bins

        # Ensure scale is not zero to avoid NaN
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = 1.0

        # Quantize to 8-bit representation
        centered_val = centered_val.to(torch.float32)
        quantized_val = (
            (centered_val / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        # Create lookup table by computing mean values for each bucket
        device = quantized_val.device
        sums = torch.zeros(self.n_bins, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_bins, dtype=torch.float32, device=device)

        sums.scatter_add_(0, quantized_val.flatten().long(), centered_val.flatten())
        counts.scatter_add_(
            0, quantized_val.flatten().long(), torch.ones_like(centered_val.flatten())
        )

        lookup = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))

        # Store quantization parameters for dequantization
        orig_dtype = val.dtype
        quant_params = (shift, scale, offset, lookup, orig_dtype)

        return quantized_val, quant_params

    @torch.no_grad()
    def _dequantize_values(self, val, quant_params):
        """
        Dequantize 8-bit values back to original representation

        Args:
            val: Quantized uint8 tensor
            quant_params: Tuple of (shift, scale, offset, lookup, orig_dtype)

        Returns:
            Dequantized tensor in original dtype
        """
        if quant_params is None:
            return val

        shift, scale, offset, lookup, orig_dtype = quant_params

        # Ensure lookup is on the same device as val
        if isinstance(lookup, torch.Tensor):
            lookup = lookup.to(val.device)

        # Convert quantized values back using lookup table
        dequantized = lookup[val.long()]

        # Apply scale and shift to get back original distribution
        val = dequantized + shift
        val = val.to(orig_dtype)

        return val


# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
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


def _idct(X, norm=None):
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


def _get_prime_divisors(n):
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


def _get_divisors(n):
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


def _get_smaller_split(n, close_to):
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n
