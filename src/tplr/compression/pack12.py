import torch


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


def unpack_12bit_indices(packed: torch.Tensor, values_shape: tuple[int, ...] ) -> torch.Tensor:
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