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


import asyncio
import math
import time
from typing import TYPE_CHECKING, TypeVar, cast

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType
from torch.nn.parallel import DistributedDataParallel as DDP

import tplr
from tplr.logging import logger

if TYPE_CHECKING:
    from neurons.aggregator import AggregationServer
    from neurons.miner import Miner
    from neurons.validator import Validator

NeuronT = TypeVar("NeuronT", "Miner", "Validator")


def prepare_gradient_dict(self, pages_info_for_metadata: list, step_window: int):
    """
    Prepares the gradient dictionary for sharing.
    This method should be called by ALL RANKS.
    It updates model parameters (weight decay) and self.momentum consistently across ranks
    using DDP-averaged gradients. The returned compressed gradient dict will be
    identical on all ranks. Rank 0 will use its copy for uploading.

    Args:
        pages_info_for_metadata (list): The pages information from the CURRENT RANK for metadata.
                                        If global pages info is needed in metadata, it must be aggregated
                                        before calling put by rank 0.
        step_window (int): The current window number.

    Returns:
        tuple: (gradient_dict, xshapes, totalks, transmitted_grads)
                These will be identical across all ranks.
    """
    gradient_output_dict = {}
    xshapes_output = {}
    totalks_output = {}
    transmitted_grads_output = {}  # For rank 0 logging or diagnostics

    # Learning rate should be consistent as scheduler is stepped by all ranks
    lr = self.scheduler.get_last_lr()[0]

    self.gradient_iteration_counter += 1
    logger.debug(  # Changed to debug for less noise from all ranks
        f"[Rank {self.rank}] prepare_gradient_dict iter_count: {self.gradient_iteration_counter}"
    )

    is_first_iteration = self.gradient_iteration_counter == 1
    is_early_iteration = self.gradient_iteration_counter <= 5

    # Get the underlying model if DDP is used
    model_to_access = self.model.module if isinstance(self.model, DDP) else self.model

    for n, p in model_to_access.named_parameters():
        if p.grad is None:
            logger.warning(
                f"[Rank {self.rank}] Param '{n}' has no gradient. Skipping in prepare_gradient_dict."
            )
            continue

        # 1. Apply weight decay directly to parameters.
        # This modification is done by all ranks identically, so parameters remain synced.
        p.data.mul_(1.0 - lr * self.hparams.weight_decay)

        # 2. Update custom momentum.
        # Ensure momentum tensor is on the same device as the parameter.
        if self.momentum[n].device != p.device:
            self.momentum[n] = self.momentum[n].to(p.device)

        self.momentum[n].mul_(self.hparams.momentum_decay)

        # p.grad is already DDP-averaged.
        # Ensure grad is on the correct device (should be by default from DDP).
        current_grad = p.grad  # No .to(p.device) needed if DDP did its job

        if is_first_iteration:
            # Set momentum directly to grad (scaled by lr)
            # Use .copy() or .clone() to avoid modifying p.grad if it's used elsewhere,
            # though after this point, optimizer.zero_grad() is usually called.
            self.momentum[n] = current_grad.clone() * lr
        else:
            # Add DDP-averaged grad to momentum
            self.momentum[n].add_(current_grad, alpha=lr)

        # 3. Compress momentum for transmission.
        # self.momentum[n] is now identical across ranks.
        # transformer.encode and compressor.compress are deterministic.
        encoded_momentum = self.transformer.encode(self.momentum[n])
        idxs, vals, xshape, totalk, quant_params = self.compressor.compress(
            encoded_momentum, self.hparams.topk_compression
        )

        # 4. Estimate transmitted gradient and adjust momentum.
        # This step also happens identically on all ranks.
        # Note: `p` passed to decompress is for shape/dtype reference.
        # The actual parameter `p` has already had weight decay applied.
        decompressed_momentum_estimate = self.transformer.decode(
            self.compressor.decompress(p, idxs, vals, xshape, totalk, quant_params)
        )

        if not is_early_iteration:
            self.momentum[n].sub_(decompressed_momentum_estimate)

        # Store compressed representation (will be identical on all ranks)
        gradient_output_dict[n + "idxs"] = idxs
        gradient_output_dict[n + "vals"] = vals
        gradient_output_dict[n + "quant_params"] = quant_params
        xshapes_output[n] = xshape
        totalks_output[n] = totalk
        transmitted_grads_output[n] = decompressed_momentum_estimate  # For diagnostics

    # Attach metadata (specific to this rank's pages_info initially)
    # If rank 0 needs global pages_info, it must gather it separately before PUT.
    gradient_output_dict["metadata"] = {
        "pages_info": pages_info_for_metadata,  # This rank's pages
        "window": step_window,
        "rank": self.rank,  # Could be useful for debugging metadata source
    }
    logger.debug(
        f"[Rank {self.rank}] Attached metadata to gradient: {gradient_output_dict['metadata']}"
    )

    return (
        gradient_output_dict,
        xshapes_output,
        totalks_output,
        transmitted_grads_output,
    )


async def update_peers(
    instance: NeuronT | "AggregationServer", window: int, peer_start: float
) -> None:
    # Check if peers list is empty and fetch previous list if needed
    if len(instance.comms.peers) == 0:
        logger.info(
            "Current peers list is empty, attempting to fetch previous peer list"
        )
        result = await instance.comms.get_peer_list(fetch_previous=True)
        if result is not None:
            prev_peers, prev_update_window = result
            logger.info(
                f"Got previous peer list with {len(prev_peers)} peers "
                f"and update window {prev_update_window}"
            )
            instance.comms.peers = prev_peers
            # Don't set next_peers here, as we want the normal update process to continue
        else:
            logger.warning(
                "Failed to fetch previous peer list, continuing with empty peers"
            )

    # Get next peers
    if (
        instance.next_peers is None  # next peers are not fetched yet
        and instance.peers_update_window  # they should be on bucket by now
        + instance.hparams.peer_replacement_frequency
        - window
        < instance.hparams.peer_list_window_margin
    ):
        result = await instance.comms.get_peer_list()
        if result is None:
            logger.info("Unable to get peer list from bucket")
        else:
            next_peers, peers_update_window = result
            logger.info(
                f"Got peer list {next_peers} and update window "
                f"{peers_update_window} from bucket"
            )
            if (
                instance.peers_update_window is None
                or peers_update_window > instance.peers_update_window
            ):
                instance.next_peers = next_peers
                instance.peers_update_window = peers_update_window
                logger.info("This list is new, updating next_peers")

    # Update peers, if it's time
    if instance.next_peers is not None and window >= instance.peers_update_window:
        instance.comms.peers = instance.next_peers
        late_text = (
            f"{window - instance.peers_update_window} windows late"
            if window - instance.peers_update_window > 0
            else "on time"
        )
        logger.info(
            f"{tplr.P(window, tplr.T() - peer_start)} Updated peers "
            f"{late_text} - gather:{len(instance.comms.peers)}. Next update "
            f"expected on step window "
            f"{instance.peers_update_window + instance.hparams.peer_list_window_margin}"
        )
        instance.next_peers = None
    else:
        reason = (
            "next peers are not defined yet"
            if instance.next_peers is None
            else f"sync window is {window} and peers update window "
            f"is {instance.peers_update_window}"
        )
        logger.info(f"Not time to replace peers: {reason}")


async def catchup_with_aggregation_server(
    instance: NeuronT, checkpoint_current_window: int
):
    """
    Catch up the model by applying aggregated gradients from the aggregation server
    and verifying against the validator's debug dict. Uses retry logic for the most
    recent window if needed.
    """
    logger.info("Starting catchup with aggregation server...")

    # Start from the checkpoint window and continue until we reach the current window
    checkpoint_window = checkpoint_current_window + 1
    target_window = instance.current_window

    logger.info(
        f"Catching up from window {checkpoint_window} to current window {target_window}"
    )

    # Apply aggregation for each step, checking for current window changes
    current_step = checkpoint_window
    while current_step < target_window:
        logger.info(
            f"\nProcessing catchup for window {current_step} (Target: {target_window})"
        )

        # Load aggregation for current window
        agg_data = await instance.comms.load_aggregation(window=current_step)

        # For the last window in catchup, we might need to retry a few times
        if agg_data is None and current_step == target_window - 1:
            max_retries = 7
            retry_count = 0
            retry_delay = 10

            logger.info(
                f"No aggregation for latest window {current_step}, will retry up to {max_retries} times"
            )

            while retry_count < max_retries and agg_data is None:
                retry_count += 1
                logger.info(
                    f"Retry {retry_count}/{max_retries} for window {current_step}"
                )
                await asyncio.sleep(retry_delay)

                # Try to load aggregation again
                agg_data = await instance.comms.load_aggregation(window=current_step)

                if agg_data is not None:
                    logger.info(
                        f"Successfully loaded aggregation on retry {retry_count}"
                    )

            if agg_data is None:
                logger.warning(
                    f"Failed to load aggregation after {max_retries} retries"
                )

        # Process the aggregation data if available
        if agg_data:
            update_start = time.time()

            # Process the loaded data
            processed_agg_data = process_loaded_data(instance.model, agg_data)

            if processed_agg_data is not None:
                # Get learning rate for this step
                lr = instance.scheduler.get_last_lr()[0]
                weight_decay = instance.hparams.weight_decay

                # Apply the gradients to the model parameters
                for name, param in instance.model.named_parameters():
                    if name in processed_agg_data["tensors"]:
                        # Apply weight decay to the parameter manually if needed
                        if weight_decay > 0:
                            with torch.no_grad():
                                param.data.mul_(1.0 - lr * weight_decay)

                        # Move aggregation tensor to device
                        agg_tensor = processed_agg_data["tensors"][name].to(
                            instance.config.device  # type: ignore
                        )

                        # Set the gradient instead of directly updating the parameter
                        if param.grad is None:
                            param.grad = agg_tensor
                        else:
                            param.grad.copy_(agg_tensor)

                logger.info(
                    f"Window {current_step} - Set gradients in {time.time() - update_start:.2f}s"
                )

                # Let the optimizer handle the parameter updates
                instance.optimizer.step()
                instance.scheduler.step()
                torch.cuda.empty_cache()

                logger.info(
                    f"Successfully applied aggregation for window {current_step}"
                )

                # Get debug dict and compare with current model parameters
                debug_dict_result = await instance.comms.get_debug_dict(current_step)
                if (
                    isinstance(debug_dict_result, dict)
                    and "state_dict" in debug_dict_result
                ):
                    debug_state_dict = cast(
                        dict[str, list[float]], debug_dict_result["state_dict"]
                    )

                    # Use our new function to compare model with debug dict
                    comparison_metrics = await compare_model_with_debug_dict(
                        model=instance.model,
                        debug_dict=debug_state_dict,
                        learning_rate=lr,
                    )

                    if comparison_metrics["success"]:
                        # Log the comparison metrics
                        logger.info(
                            f"Window {current_step} - L2 norm difference between model and debug values: "
                            f"{comparison_metrics['l2_norm']}"
                        )
                        logger.info(
                            f"Window {current_step} - Average L2 norm per parameter: "
                            f"{comparison_metrics['avg_l2_norm']}"
                        )
                        logger.info(
                            f"Window {current_step} - Average absolute difference per parameter: "
                            f"{comparison_metrics['avg_abs_diff']}"
                        )
                        logger.info(
                            f"Window {current_step} - Average steps behind: "
                            f"{comparison_metrics['avg_steps_behind']}"
                        )
                    else:
                        logger.warning(
                            f"Failed to compare model with debug dict for window {current_step}"
                        )
                else:
                    logger.warning(
                        f"Invalid debug dict format for window {current_step}"
                    )
            else:
                logger.warning(
                    f"Failed to process aggregation data for window {current_step}"
                )
                # Still advance the optimizer and scheduler
                instance.optimizer.step()
                instance.scheduler.step()
        else:
            logger.warning(f"No aggregation data found for window {current_step}")
            # Don't advance the optimizer and scheduler

        # Update global step and move to next window
        instance.global_step = current_step - instance.start_window
        current_step += 1

        # Check if current_window has changed during processing
        if instance.current_window > target_window:
            target_window = instance.current_window
            logger.info(
                f"Current window advanced during catchup, new target: {target_window}"
            )

    # Update global step after catchup
    instance.global_step = target_window - instance.start_window
    logger.info(f"Catchup complete. Global step updated to {instance.global_step}")


def process_loaded_data(model: torch.nn.Module, compressed_data: dict) -> dict | None:
    """
    Unpack the compressed tensor data from the aggregation server.

    Args:
        compressed_data: The compressed tensor data

    Returns:
        Dictionary with unpacked tensors
    """
    state_dict = compressed_data.get("state_dict")
    if state_dict is None:
        return None

    result = {
        "timestamp": state_dict.get("timestamp", None),
        "window": state_dict.get("window", None),
        "version": state_dict.get("version", None),
        "tensors": {},
    }

    for name, param in model.named_parameters():
        if name in state_dict:
            original_shape = param.shape
            # Use unpack_binary_tensor from the sample, but in our context
            unpacked = unpack_binary_tensor(state_dict[name], original_shape)
            result["tensors"][name] = unpacked
            logger.debug(f"Unpacked tensor {name} with shape {original_shape}")

    logger.info(f"Successfully unpacked {len(result['tensors'])} tensors")
    return result


async def compare_model_with_debug_dict(
    model: nn.Module,
    debug_dict: dict[str, list[float]],
    learning_rate: float,
    index_range: tuple[int, int] = (0, 2),
) -> dict[str, bool | float | int]:
    """
    Compares a model's parameters with a debug dictionary to measure synchronization.

    Args:
        model: The PyTorch model with parameters to compare
        debug_dict: Debug dictionary containing parameter debug values
        learning_rate: Current learning rate to normalize differences

    Returns:
        dict: Comparison metrics including L2 norm, absolute differences, and steps behind measurements
    """
    # Initialize metrics
    total_squared_diff = 0.0
    total_abs_diff = 0.0
    param_count = 0
    max_diff = 0.0

    # Compare each parameter with its debug entry
    for name, param in model.named_parameters():
        debug_key = name + "_debug"

        if debug_key in debug_dict and isinstance(debug_dict[debug_key], list):
            # Get the parameter values (first two elements to match debug dict)
            param_data = param.data.flatten()[
                index_range[0] : index_range[1]
            ].detach()  # Keep on device

            # Convert debug data to tensor on the same device
            debug_data = torch.tensor(
                debug_dict[debug_key], device=param.device, dtype=param.dtype
            )

            # Compute differences
            diffs = param_data - debug_data
            squared_diff = torch.sum(diffs**2).item()
            abs_diff = torch.abs(diffs).sum().item()
            max_param_diff = torch.max(torch.abs(diffs)).item()

            # Update totals
            total_squared_diff += squared_diff
            total_abs_diff += abs_diff
            param_count += param_data.numel()
            max_diff = max(max_diff, max_param_diff)

    # Calculate final metrics
    l2_norm = torch.sqrt(torch.tensor(total_squared_diff)).item()
    avg_l2_norm = l2_norm / param_count if param_count > 0 else math.inf
    avg_abs_diff = total_abs_diff / param_count if param_count > 0 else math.inf

    # Normalize differences by learning rate to get "steps behind" metric
    avg_steps_behind = avg_abs_diff / learning_rate if learning_rate > 0 else math.inf
    max_steps_behind = max_diff / learning_rate if learning_rate > 0 else math.inf

    # Prepare return metrics
    metrics: dict[str, bool | float | int] = {
        "success": True,
        "l2_norm": l2_norm,
        "avg_l2_norm": avg_l2_norm,
        "avg_abs_diff": avg_abs_diff,
        "max_diff": max_diff,
        "avg_steps_behind": avg_steps_behind,
        "max_steps_behind": max_steps_behind,
        "param_count": param_count,
        "learning_rate": learning_rate,
    }

    return metrics


def unpack_binary_tensor(packed_tensor: torch.Tensor, original_shape: torch.Size):
    """
    Unpack a 1-bit representation tensor back to ±1 values.

    Args:
        packed_tensor: The packed binary tensor
        original_shape: The original shape of the tensor

    Returns:
        Unpacked tensor with original shape
    """
    total_elements = int(torch.prod(torch.tensor(original_shape)).item())

    # Create a flat tensor to hold the unpacked values
    unpacked = torch.zeros(total_elements, dtype=torch.float32)

    for i in range(8):
        mask = 1 << i
        bits = (packed_tensor & mask) >> i
        # Convert 0/1 to -1/+1
        unpacked[i::8] = (bits.float() * 2) - 1

    return unpacked.reshape(original_shape)


# Function to pack signed weights into 1-bit representation
def pack_binary_tensor(tensor: torch.Tensor, device: DeviceLikeType):
    """Pack a tensor of +1/-1 values into a compact binary representation."""
    tensor = (tensor > 0).to(torch.uint8)  # Convert +1 to 1, -1 to 0
    tensor = tensor.view(-1)  # Flatten
    packed_tensor = torch.zeros(
        (tensor.shape[0] + 7) // 8, dtype=torch.uint8, device=device
    )

    for i in range(8):
        packed_tensor |= tensor[i::8] << i  # Pack 8 values per byte

    return packed_tensor
