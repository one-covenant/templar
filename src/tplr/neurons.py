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
import gc
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor as DT
from torch.distributed.tensor import distribute_tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from wandb.sdk.wandb_run import Run

import tplr
from tplr.compress import unpack_12bit_indices
from tplr.distributed import dist_helper

if TYPE_CHECKING:
    from neurons.miner import Miner
    from neurons.validator import Validator

NeuronT = TypeVar("NeuronT", "Miner", "Validator")


def _resolve_cuda_device(device: str | torch.device | None) -> torch.device | None:
    """Return a torch.device for CUDA operations if available."""
    if not torch.cuda.is_available():
        return None

    if device is None:
        try:
            current_index = torch.cuda.current_device()
        except RuntimeError:
            if torch.cuda.device_count() == 0:
                return None
            return torch.device("cuda:0")
        return torch.device(f"cuda:{current_index}")

    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        if torch.cuda.device_count() == 0:
            return None
        return torch.device("cuda:0")

    return resolved if resolved.type == "cuda" else None


def prepare_gradient_dict(miner: "Miner", step_window: int, null_round: bool = False):
    """
    DTensor-deadlock-safe:
    - All ranks: rendezvous on DTensor grads (GFULL) and DTensor params (PFULL).
    - Only owning ranks: momentum update, encode, compress, estimate/decode, EF update.

    Args:
        miner: Miner instance containing model, compressor, transformer, etc.
        step_window: Current window number
        null_round: If True, this is a null/warmup round and error feedback should be cleared
    """

    # ------------ helpers ------------
    def ddp_initialized():
        return dist.is_available() and dist.is_initialized()

    def is_dtensor(x):
        try:
            from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]

            return isinstance(x, DTensor)
        except Exception:
            return type(x).__name__ in {"DTensor", "DistributedTensor", "DT"}

    def get_mesh_group(x):
        if not is_dtensor(x):
            return None
        mesh = getattr(x, "device_mesh", None)
        if mesh is None:
            spec = getattr(x, "_spec", None)
            mesh = getattr(spec, "mesh", None)
        if mesh is not None:
            try:
                return mesh.get_group()
            except Exception:
                pass
        return dist.group.WORLD if ddp_initialized() else None

    def barrier(group=None):
        if ddp_initialized() and group is not None:
            dist.barrier(group=group)

    # ------------ start ------------
    gradient, xshapes, totalks = {}, {}, {}
    use_dct = getattr(miner.hparams, "use_dct", False)
    topk = getattr(miner.hparams, "topk_compression", 32)

    if isinstance(miner.model, torch.nn.parallel.DistributedDataParallel):
        model_iterator = miner.model.module.named_parameters()
    else:
        model_iterator = miner.model.named_parameters()

    # Build params dict once to avoid repeated iteration
    params_dict = dict(miner.model.named_parameters())

    # Batch load all error feedback tensors to GPU
    for n in miner.owned_params:
        if miner.error_feedback.get(n, None) is not None:
            if miner.error_feedback[n].is_cuda:
                continue
            # Get the device from the corresponding parameter
            param = params_dict.get(n)
            if param is not None:
                miner.error_feedback[n] = miner.error_feedback[n].to(
                    param.device, non_blocking=True
                )

    for _, (n, p) in enumerate(model_iterator, 1):
        owned = n in miner.owned_params
        p_is_dt = is_dtensor(p)
        g = getattr(p, "grad", None)
        g_is_dt = is_dtensor(g)

        # --- 1) Grad full_tensor rendezvous (GFULL) ---
        if g_is_dt:
            grp_g = get_mesh_group(g)
            barrier(grp_g)
            assert g is not None
            grad_full = g.full_tensor().to(p.device)
            barrier(grp_g)
        else:
            if g is None and not p_is_dt:
                continue
            assert g is not None, f"p.grad is None for {n}"
            grad_full = g.to(p.device)

        # Non-owners: after participating in grad collective, drop grad and continue.
        if not owned:
            p.grad = None
            continue

        # --- 3) Momentum buffer update (owner only) ---
        # Handle DTensor error feedback by creating new regular tensor if needed
        error_feedback = miner.error_feedback[n]
        if error_feedback is None:
            error_feedback = torch.zeros_like(grad_full, device=p.device)
        elif error_feedback.device != p.device:
            # Should already be on GPU from batch load, but handle edge cases
            error_feedback = error_feedback.to(p.device)

        # Clear error feedback during null rounds to prevent accumulation of invalid gradients
        if null_round:
            error_feedback.zero_()
        else:
            error_feedback.mul_(miner.hparams.momentum_decay)
            error_feedback.add_(grad_full)

        # --- 4) Encode & compress (owner only) ---
        encoded = miner.transformer.encode(error_feedback, use_dct=use_dct)

        idxs, vals, xshape, totalk, quant_params = miner.compressor.compress(
            encoded, topk
        )
        del encoded

        # --- 5) Decompress reference (owner only) ---
        # Pass p directly - decompress only uses p.device and p.dtype
        decompressed = miner.compressor.decompress(
            p, idxs, vals, xshape, totalk, quant_params
        )

        # --- 6) Decode & error-feedback update (owner only) ---
        transmit_grad = miner.transformer.decode(decompressed, use_dct=use_dct)
        del decompressed
        error_feedback.sub_(transmit_grad)
        # Keep error feedback on GPU for now, batch offload later
        miner.error_feedback[n] = error_feedback
        del transmit_grad, error_feedback

        # --- 7) Pack outputs (move compressed artifacts to CPU asynchronously) ---
        # Using non_blocking=True for async D2H transfers when CUDA is available
        if isinstance(idxs, torch.Tensor):
            if torch.cuda.is_available():
                cpu_idxs = torch.empty_like(idxs, device="cpu", pin_memory=True)
                cpu_idxs.copy_(idxs, non_blocking=True)
                gradient[n + "idxs"] = cpu_idxs
            else:
                gradient[n + "idxs"] = idxs.cpu()
        else:
            gradient[n + "idxs"] = idxs

        if isinstance(vals, torch.Tensor):
            if torch.cuda.is_available():
                cpu_vals = torch.empty_like(vals, device="cpu", pin_memory=True)
                cpu_vals.copy_(vals, non_blocking=True)
                gradient[n + "vals"] = cpu_vals
            else:
                gradient[n + "vals"] = vals.cpu()
        else:
            gradient[n + "vals"] = vals
        gradient[n + "quant_params"] = quant_params
        xshapes[n] = xshape
        totalks[n] = totalk

        # Clear per-param grad
        p.grad = None

    # Batch offload all error feedback tensors to CPU with pinned memory
    for name in miner.error_feedback:
        if (
            miner.error_feedback[name] is not None
            and miner.error_feedback[name].is_cuda
        ):
            # Copy to the pre-allocated pinned buffer
            miner.error_feedback_cpu_buffers[name].copy_(
                miner.error_feedback[name], non_blocking=True
            )
            miner.error_feedback[name] = miner.error_feedback_cpu_buffers[name]

    # Single synchronization at the end for all async operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gradient["metadata"] = {"window": step_window}
    return gradient, xshapes, totalks


@torch.no_grad()
def outer_step(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    gather_result: SimpleNamespace | None,
    transformer: tplr.compress.ChunkingTransformer,
    compressor: tplr.compress.TopKCompressor,
    xshapes: dict,
    totalks: dict,
    device: str,
    is_master: bool,
    world_size: int,
    use_dct: bool = False,
    wandb_run: Run | None = None,
    global_step: int | None = None,
) -> dict | None:
    """
    Apply aggregated gradients from peers to the model.

    Distributed gather merges partials internally, so this function always
    receives a single merged result (not a list of partials).

    Memory-minimizing implementation:
      - Builds and applies ONE param's grad at a time
      - Calls optimizer.step() per param (others have grad=None, so they're skipped)
      - Frees all temporaries and grad immediately after each step

    Returns:
      Fingerprint dict containing gradient statistics (master rank only), or None.
    """
    return _outer_step_single(
        model=model,
        optimizer=optimizer,
        gather_result=gather_result,
        transformer=transformer,
        compressor=compressor,
        xshapes=xshapes,
        totalks=totalks,
        device=device,
        is_master=is_master,
        world_size=world_size,
        use_dct=use_dct,
        wandb_run=wandb_run,
        global_step=global_step,
    )


def _outer_step_single(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    gather_result: SimpleNamespace | None,
    transformer: tplr.compress.ChunkingTransformer,
    compressor: tplr.compress.TopKCompressor,
    xshapes: dict,
    totalks: dict,
    device: str,
    is_master: bool,
    world_size: int,
    use_dct: bool = False,
    wandb_run: Run | None = None,
    global_step: int | None = None,
) -> dict | None:
    """
    Original single-result outer step implementation.

    Memory-minimizing variant:
      - Builds and applies ONE param's grad at a time.
      - Calls optimizer.step() per param (others have grad=None, so they're skipped).
      - Frees all temporaries and grad immediately after each step.

    Returns:
      Fingerprint dict containing gradient statistics (master rank only), or None.
    """
    model.train()

    # Free any existing grads entirely (do not allocate zeros)
    optimizer.zero_grad(set_to_none=True)

    ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    src_rank = 0
    on_src = is_master or not ddp
    cuda_device = _resolve_cuda_device(device)

    # Only master reads aggregated payload (others rely on broadcasts).
    # Accept both SimpleNamespace and plain dict payloads.
    src_sd: dict | None = None
    if (
        on_src
        and gather_result is not None
        and getattr(gather_result, "state_dict", None) is not None
    ):
        sd = gather_result.state_dict
        src_sd = vars(sd).copy() if isinstance(sd, SimpleNamespace) else dict(sd)

    # compact flag broadcast
    def _bcast_flag(v: int) -> int:
        t = torch.tensor([v], device=device, dtype=torch.int32)
        if ddp:
            dist.broadcast(t, src_rank)
        return int(t.item())

    # optional stats
    min_median_norm = float("inf")
    max_median_norm = float("-inf")

    # Initialize fingerprint accumulator (master rank only)
    fingerprint: dict | None = None
    if on_src:
        fingerprint = {
            "param_norms": {},
            "param_means": {},
            "total_norm_sq": 0.0,
            "total_elements": 0,
        }

    def _idx_to_device(obj, dev: str):
        """
        Move indices to device, supporting:
          • Tensor
          • (packed_tensor, original_shape) for 12-bit packed indices
          • nested list/tuple containers of the above
        We only move the tensor parts; shapes/ints stay on CPU.
        """
        if torch.is_tensor(obj):
            return obj.to(device=dev, non_blocking=True)
        if isinstance(obj, tuple) and len(obj) == 2 and torch.is_tensor(obj[0]):
            return (obj[0].to(device=dev, non_blocking=True), obj[1])
        if isinstance(obj, list):
            return [_idx_to_device(x, dev) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_idx_to_device(x, dev) for x in obj)
        return obj

    for name, p in model.named_parameters():
        # ---- master decides if this param has an update; others receive a flag ----
        has_update = 0
        payload = None

        if on_src and src_sd is not None:
            idxs = src_sd.get(name + "idxs")
            vals = src_sd.get(name + "vals")
            qps = src_sd.get(name + "quant_params")

            if idxs is not None and vals is not None:
                if not isinstance(idxs, (list, tuple)):
                    idxs = [idxs]
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                # Dequantize values directly on target device (H2D per-block if needed)
                vals_f32 = compressor.maybe_dequantize_values(vals, qps, device)
                if vals_f32:
                    # Ensure indices (or packed tuples) live on the same device as 'ref'
                    idxs_dev = _idx_to_device(idxs, device)
                    payload = (idxs_dev, vals_f32)
                    has_update = 1

        flag_result = _bcast_flag(has_update)
        if flag_result == 0:
            # Nothing to apply for this param
            continue

        full_grad_src = torch.empty(1)
        decompressed = None
        block_norms = None

        # ------- build the full dense grad on the source rank only -------
        if on_src:
            try:
                idxs_dev, vals_f32 = payload  # type: ignore[misc]
                # Per-block norms for stats/optional clipping inside batch_decompress
                block_norms = torch.stack([torch.norm(v, p=2) for v in vals_f32])

                # stats
                med = float(torch.median(block_norms).item())
                min_median_norm = min(min_median_norm, med)
                max_median_norm = max(max_median_norm, med)

                # Use empty_like to avoid copying the param; just provide dtype/device/shape
                ref = torch.empty_like(p, device=device, dtype=p.dtype)
                decompressed = compressor.batch_decompress(
                    ref,
                    idxs_dev,
                    vals_f32,
                    xshapes[name],
                    totalks[name],
                    quantize_params=None,
                    block_norms=block_norms,
                    normalise=False,
                    clip_norm=True,
                )

                full_grad_src = transformer.decode(decompressed, use_dct=use_dct)
                # Single conversion to target dtype+device to avoid extra temporaries
                full_grad_src = full_grad_src.to(
                    dtype=p.dtype, device=p.device, non_blocking=True
                )

                # Accumulate fingerprint statistics for this parameter
                if fingerprint is not None:
                    param_norm = torch.norm(full_grad_src, p=2).item()
                    fingerprint["param_norms"][name] = param_norm
                    fingerprint["total_norm_sq"] += param_norm**2
                    fingerprint["total_elements"] += full_grad_src.numel()
                    fingerprint["param_means"][name] = full_grad_src.mean().item()
            finally:
                # Free intermediate pieces ASAP (existence-guarded)
                try:
                    del decompressed
                except UnboundLocalError:
                    pass
                # vals/idxs/qps live in src_sd; only local views should be dropped
                try:
                    del vals_f32, idxs_dev, block_norms, ref
                except UnboundLocalError:
                    pass

        # ------- distribute/broadcast directly into p.grad, step, then free -------
        if isinstance(p, DT):
            # DTensor param: scatter shards from master
            src_tensor = (
                full_grad_src
                if on_src
                else torch.empty(p.shape, device=p.device, dtype=p.dtype)
            )
            new_grad = distribute_tensor(
                src_tensor,
                device_mesh=p.device_mesh,
                placements=p.placements,
                src_data_rank=src_rank,
            )
            # master no longer needs the full dense grad
            if on_src:
                del full_grad_src
                full_grad_src = None

            # quick sanity (view, no extra big alloc)
            local_view = new_grad.to_local()
            if not torch.isfinite(local_view).all():
                del new_grad, local_view
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            p.grad = new_grad  # DTensor grad
            del new_grad, local_view

        else:
            # Replicated param: broadcast dense grad once.
            if ddp:
                if on_src:
                    # Broadcast from the source tensor; then reuse it as grad
                    dist.broadcast(full_grad_src, src_rank)  # type: ignore[arg-type]
                    p.grad = full_grad_src
                    full_grad_src = None
                else:
                    # Receive directly into p.grad to avoid an extra buffer
                    p.grad = torch.empty_like(p, device=p.device, dtype=p.dtype)
                    dist.broadcast(p.grad, src_rank)  # type: ignore[arg-type]
            else:
                # Single process: just use the built tensor
                p.grad = full_grad_src
                full_grad_src = None

            if p.grad is not None and not torch.isfinite(p.grad).all():  # type: ignore[arg-type]
                p.grad = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        # ---- apply update immediately for THIS param and free its grad ----
        # ---- apply update immediately for THIS param and free its grad ----
        optimizer.step()
        p.grad = None  # free grad storage right away
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # optional W&B (master only)
    if (
        on_src
        and wandb_run is not None
        and global_step is not None
        and max_median_norm > float("-inf")
    ):
        wandb_run.log(
            {
                "compress/min_median_block_norm": min_median_norm,
                "compress/max_median_block_norm": max_median_norm,
            },
            step=global_step,
        )

    # Extra safety: ensure no grads are left allocated
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute final fingerprint (master rank only)
    if on_src and fingerprint is not None:
        fingerprint["global_l2_norm"] = math.sqrt(fingerprint["total_norm_sq"])
        return fingerprint
    return None


async def update_peers(instance: NeuronT, window: int, peer_start: float) -> None:
    # Check if peers list is empty and fetch previous list if needed
    if len(instance.comms.peers) == 0:
        tplr.logger.info(
            "Current peers list is empty, attempting to fetch previous peer list"
        )
        result = await instance.comms.get_peer_list(fetch_previous=True)
        if result is not None:
            prev_peers, prev_reserve, prev_update_window = result
            tplr.logger.info(
                f"Got previous peer list with {len(prev_peers)} peers "
                f"and update window {prev_update_window}"
            )
            instance.comms.peers = prev_peers
            instance.comms.reserve_peers = prev_reserve

            # Don't set next_peers here, as we want the normal update process to continue
        else:
            tplr.logger.warning(
                "Failed to fetch previous peer list, continuing with empty peers"
            )

    # Get next peers
    if (
        instance.next_peers is None  # next peers are not fetched yet
        and instance.peers_update_window  # they should be on bucket by now
        + instance.hparams.peer_replacement_frequency
        - window
        <= instance.hparams.peer_list_window_margin
    ):
        result = await instance.comms.get_peer_list()
        if result is None:
            tplr.logger.info("Unable to get peer list from bucket")
        else:
            next_peers, reserve_peers, peers_update_window = result
            tplr.logger.info(
                f"Got peer list {next_peers} and update window "
                f"{peers_update_window} from bucket"
            )
            if (
                instance.peers_update_window is None
                or peers_update_window > instance.peers_update_window
            ):
                instance.next_peers = next_peers
                instance.next_reserve_peers = reserve_peers
                instance.peers_update_window = peers_update_window
                tplr.logger.info("This list is new, updating next_peers")

    # Update peers, if it's time
    if instance.next_peers is not None and window >= instance.peers_update_window:
        # ── atomic switch ─────────────────────────────────────────────
        instance.comms.peers = instance.next_peers
        instance.comms.reserve_peers = (
            instance.next_reserve_peers
            if instance.next_reserve_peers is not None
            else []
        )
        late_text = (
            f"{window - instance.peers_update_window} windows late"
            if window - instance.peers_update_window > 0
            else "on time"
        )
        tplr.logger.info(
            f"{tplr.P(window, tplr.T() - peer_start)} Updated peers "
            f"{late_text} - gather:{len(instance.comms.peers)}, "
            f"reserve:{len(instance.comms.reserve_peers)}. Next update "
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
        tplr.logger.info(f"Not time to replace peers: {reason}")


async def load_checkpoint_with_fallback(
    instance: NeuronT,
) -> tuple[bool, int, int, bool]:
    """
    Load checkpoint with fallback logic.

    1. First try loading from current version
    2. If not found, try bootstrap version if configured
    3. Return checkpoint status and metadata

    Returns:
        tuple of (checkpoint_ok, checkpoint_window, global_step, from_bootstrap)
    """
    ckpt_ok = False
    ckpt_sync_win = 0
    ckpt_global_step = 0
    from_bootstrap = False

    # First check if current version has any checkpoints
    latest_current_window = await instance.ckpt._discover_latest(
        prefer_highest_staked=True
    )

    if latest_current_window is not None:
        # Current version checkpoint exists, load it
        res = await instance.ckpt.download_and_load(
            model=instance.model,
            window=latest_current_window,
            shared_fs=True,
            process_group=None,
            prefer_highest_staked=True,
        )
        if res is not None:
            ckpt_ok = True
            ckpt_sync_win, ckpt_global_step = res
            instance.model_initialized = True  # Model now has real weights
            tplr.logger.info(
                f"Loaded current version checkpoint (window={ckpt_sync_win}, "
                f"global_step={ckpt_global_step})"
            )

    # If no current version checkpoint and bootstrap is configured, try that
    if not ckpt_ok and instance.bootstrap_version:
        tplr.logger.info(
            f"No current version checkpoint found, trying bootstrap version "
            f"{instance.bootstrap_version}"
        )
        # Try specific window if configured, otherwise latest
        bootstrap_window = getattr(instance.hparams, "checkpoint_init_window", None)
        bootstrap_ckpt = tplr.DCPCheckpointer(
            instance.comms,
            uid=instance.uid,
            version=instance.bootstrap_version,
            repo_root=".",
        )

        # If no specific window configured, discover latest in bootstrap version
        if bootstrap_window is None:
            bootstrap_window = await bootstrap_ckpt._discover_latest(
                prefer_highest_staked=True
            )

        if bootstrap_window is not None:
            res = await bootstrap_ckpt.download_and_load(
                model=instance.model,
                window=bootstrap_window,
                shared_fs=True,
                process_group=None,
                prefer_highest_staked=True,
            )
            if res is not None:
                ckpt_ok = True
                from_bootstrap = True
                ckpt_sync_win, ckpt_global_step = res
                instance.model_initialized = True  # Model now has real weights
                tplr.logger.info(
                    f"Loaded bootstrap checkpoint (version={instance.bootstrap_version}, "
                    f"window={ckpt_sync_win}, global_step={ckpt_global_step})"
                )

    # Handle global_step calculation if needed
    if ckpt_ok and ckpt_global_step == -1:
        if from_bootstrap:
            # For bootstrap checkpoints, try to get the start_window from that version
            bootstrap_start_window = await instance.comms.get_start_window(
                version=instance.bootstrap_version
            )
            if bootstrap_start_window is not None:
                ckpt_global_step = ckpt_sync_win - bootstrap_start_window
                tplr.logger.info(
                    f"Bootstrap checkpoint has no global_step, calculated as {ckpt_global_step} "
                    f"(window {ckpt_sync_win} - bootstrap start {bootstrap_start_window})"
                )
            else:
                # Fallback if we can't get bootstrap start_window
                ckpt_global_step = 0
                tplr.logger.info(
                    f"Bootstrap checkpoint has no global_step and couldn't fetch bootstrap start_window, "
                    f"setting to 0 (will be corrected during catch-up)"
                )
        else:
            # For current version checkpoints, calculate from window difference
            ckpt_global_step = ckpt_sync_win - instance.start_window
            tplr.logger.info(
                f"No global_step in checkpoint, calculated as {ckpt_global_step} "
                f"(window {ckpt_sync_win} - start {instance.start_window})"
            )

    if ckpt_ok:
        instance.global_step = ckpt_global_step

    return ckpt_ok, ckpt_sync_win, ckpt_global_step, from_bootstrap


async def handle_checkpoint_catchup(
    instance: NeuronT,
    ckpt_ok: bool,
    ckpt_sync_win: int,
    ckpt_global_step: int,
    from_bootstrap: bool,
    aggregator_device: str | None = None,
) -> None:
    """
    Handle catch-up logic after checkpoint loading and replay scheduler steps.

    Args:
        instance: Miner or Validator instance
        ckpt_ok: Whether a checkpoint was successfully loaded
        ckpt_sync_win: Window number from checkpoint
        ckpt_global_step: Global step from checkpoint
        from_bootstrap: Whether checkpoint was from bootstrap version
        aggregator_device: which device to load aggregation results to
    """
    # Check for anneal mode - handles scheduler replay differently
    anneal_config = getattr(instance.hparams, "anneal_mode", {})
    anneal_enabled = anneal_config.get("enabled", False)

    if anneal_enabled:
        # Anneal mode: Calculate progress from start_global_step, not from 0
        anneal_start_global_step = anneal_config.get("start_global_step", 0)
        warmup_inner_steps = anneal_config.get("warmup_inner_steps", 100)

        # Calculate how far into anneal based on actual outer steps taken
        anneal_outer_steps = max(0, ckpt_global_step - anneal_start_global_step)
        anneal_inner_steps = anneal_outer_steps * instance.hparams.inner_steps

        # Set warmup tracking for anneal mode
        instance.warmup_inner_steps = warmup_inner_steps
        instance.warmup_steps_taken = min(anneal_inner_steps, warmup_inner_steps)

        # Replay anneal scheduler steps (NOT the old pre-anneal steps)
        instance.inner_scheduler_step_count = 0
        if (
            anneal_inner_steps > 0
            and getattr(instance, "inner_scheduler", None) is not None
        ):
            for _ in range(anneal_inner_steps):
                instance.inner_scheduler.step()
                instance.inner_scheduler_step_count += 1

        tplr.logger.info(
            f"Anneal mode: start_global_step={anneal_start_global_step}, "
            f"ckpt_global_step={ckpt_global_step}, "
            f"replayed {anneal_inner_steps} anneal scheduler steps"
        )

        # Still need to do catchup for gradients
        if not ckpt_ok:
            tplr.logger.info("No checkpoint found, will catch up from start_window")
            await catchup_with_aggregation_server(
                instance, instance.start_window, aggregator_device=aggregator_device
            )
        elif ckpt_sync_win < instance.current_window:
            catch_up_start = max(ckpt_sync_win, instance.start_window)
            tplr.logger.info(
                f"Checkpoint at window {ckpt_sync_win} is behind current {instance.current_window}, "
                f"catching up from {catch_up_start}"
            )
            await catchup_with_aggregation_server(
                instance, catch_up_start, aggregator_device=aggregator_device
            )
        else:
            tplr.logger.info(
                f"Checkpoint at window {ckpt_sync_win} is up to date with current window "
                f"{instance.current_window}"
            )
        return

    # Normal (non-anneal) mode below
    # Determine scheduler config and warmup settings
    optimizer_cfg = getattr(instance.hparams, "optimizer", {})
    opt_type = optimizer_cfg.get("type", "adamw").lower()
    opt_cfg = optimizer_cfg.get(opt_type, {})
    scheduler_cfg = opt_cfg.get("scheduler", {})

    default_warmup_inner = scheduler_cfg.get(
        "warmup_inner_steps", getattr(instance, "warmup_inner_steps", 0)
    )
    startup_warmup_inner = scheduler_cfg.get(
        "initial_warmup_inner_steps", default_warmup_inner
    )

    # Set warmup length:
    # - If resuming from bootstrap, use the longer startup warmup.
    # - If resuming from a regular checkpoint, use the default.
    # - If global_step is 0, leave as-is; the scheduler's own warmup covers this.
    if ckpt_global_step == 0:
        tplr.logger.info("Global step is 0; leaving warmup settings unchanged.")
    elif from_bootstrap:
        instance.warmup_inner_steps = startup_warmup_inner
        tplr.logger.info(
            f"Applying startup warmup_inner_steps={startup_warmup_inner} (bootstrap resume)"
        )
        instance.warmup_steps_taken = 0
    else:
        instance.warmup_inner_steps = default_warmup_inner
        tplr.logger.info(
            f"Applying resumed warmup_inner_steps={default_warmup_inner} (checkpoint resume)"
        )
        instance.warmup_steps_taken = 0

    # Decide catch-up windows and run catch-up on ALL ranks
    # When loading from bootstrap, we always need to catch up from start_window
    # to ensure we're using current version's gradients
    if not ckpt_ok:
        # No checkpoint found, catch up from start_window
        tplr.logger.info("No checkpoint found, will catch up from start_window")
        await catchup_with_aggregation_server(
            instance, instance.start_window, aggregator_device=aggregator_device
        )
    elif from_bootstrap:
        # Loading from bootstrap, catch up from start_window with current version gradients
        tplr.logger.info(
            f"Loaded bootstrap checkpoint, catching up from start_window "
            f"{instance.start_window} to {instance.current_window}"
        )
        await catchup_with_aggregation_server(
            instance, instance.start_window, aggregator_device=aggregator_device
        )
    elif ckpt_sync_win < instance.current_window:
        # Current version checkpoint is behind, catch up from checkpoint window
        catch_up_start = max(ckpt_sync_win, instance.start_window)
        tplr.logger.info(
            f"Checkpoint at window {ckpt_sync_win} is behind current {instance.current_window}, "
            f"catching up from {catch_up_start}"
        )
        await catchup_with_aggregation_server(
            instance, catch_up_start, aggregator_device=aggregator_device
        )
    else:
        tplr.logger.info(
            f"Checkpoint at window {ckpt_sync_win} is up to date with current window "
            f"{instance.current_window}"
        )

    # Replay scheduler steps based on windows completed from checkpoint
    # ckpt_global_step tracks windows, scheduler needs inner_steps per window
    total_inner_steps = ckpt_global_step * instance.hparams.inner_steps

    # Apply configurable rewind before replaying scheduler to give slack on restarts
    rewind_inner_steps = scheduler_cfg.get("replay_rewind_inner_steps", 0)
    if rewind_inner_steps > 0:
        total_inner_steps = max(total_inner_steps - rewind_inner_steps, 0)
        tplr.logger.info(
            f"Rewinding scheduler replay by {rewind_inner_steps} inner steps; "
            f"{total_inner_steps} steps remain to replay"
        )

    if total_inner_steps > 0 and getattr(instance, "inner_scheduler", None) is not None:
        for _ in range(total_inner_steps):
            # Respect flatten window during replay
            if not instance.should_skip_scheduler_step():
                instance.inner_scheduler.step()
            instance.inner_scheduler_step_count += 1
        tplr.logger.info(
            f"Replayed {total_inner_steps} scheduler steps (checkpoint global_step="
            f"{ckpt_global_step} * {instance.hparams.inner_steps} inner_steps)"
        )


async def catchup_with_aggregation_server(
    instance: NeuronT,
    checkpoint_current_window: int,
    aggregator_device: str | None = None,
) -> None:
    """
    Synchronise the local model with the chain with memory optimizations.

    For every window between the checkpoint and the current chain head:

    1. **Primary path** – download the pre-computed `aggregated_gradients`
       object uploaded by the *leader* validator and apply it via
       `tplr.neurons.outer_step`.

    2. **Fallback for the final window only** – if the leader has not yet
       published an aggregator object for `target_window - 1`, perform a live
       `instance.comms.gather( ..., key="gradient", ... )` against the current
       peer-set and apply those gradients instead.

    After each application we advance the inner LR scheduler, aggressively clear
    memory including CUDA cache and CPU memory via garbage collection.

    The loop exits when `start_w` has caught up with `instance.current_window`
    (taking into account that the chain head may advance while we are replaying).
    """
    tplr.logger.info(
        "Starting catch‑up using aggregated_gradients with memory optimization..."
    )
    assert instance.start_window is not None

    # Use provided device or default to instance's device
    catchup_device = (
        aggregator_device if aggregator_device is not None else instance.config.device
    )

    def log_memory_usage(prefix: str):
        """Log current memory usage statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            tplr.logger.info(
                f"{prefix} - GPU Memory: Allocated={allocated:.2f}GB, "
                f"Reserved={reserved:.2f}GB, Max={max_memory:.2f}GB"
            )

    leader_uid: int = instance.comms.metagraph.S.argmax().item()

    start_w = checkpoint_current_window + 1
    target_w = instance.current_window
    tplr.logger.info(f"Replaying windows {start_w} ... {target_w - 1}")

    # Log initial memory state
    log_memory_usage("Initial memory state")

    # Verify checkpoint loaded correctly before applying any gradients
    if checkpoint_current_window > 0 and instance.is_master:
        tplr.logger.info(
            f"Verifying checkpoint state at window {checkpoint_current_window}"
        )
        debug_fetch = await instance.comms.get(
            uid=str(leader_uid),
            window=checkpoint_current_window,
            key="debug",
            local=False,
            stale_retention=10,
        )

        if debug_fetch.success and isinstance(debug_fetch.data, dict):
            debug_dict = debug_fetch.data  # validator's payload

            cmp = await compare_model_with_debug_dict(
                instance.model,
                debug_dict,
                param_avg_change={},  # Empty since we haven't started tracking yet
                learning_rate=instance.hparams.learning_rate,
            )
            if cmp["success"]:
                tplr.logger.info(
                    f"✓ Checkpoint verification: model matches window {checkpoint_current_window} "
                    f"(l2_norm={cmp['l2_norm']:.4f}, avg_steps_behind={cmp['avg_steps_behind']:.3f})"
                )
                if cmp["l2_norm"] > 0.1:  # Threshold for acceptable difference
                    tplr.logger.warning(
                        f"⚠️ Large L2 norm difference detected: {cmp['l2_norm']:.4f}. "
                        f"Checkpoint may not have loaded correctly."
                    )
            else:
                tplr.logger.warning(
                    f"⚠️ Could not verify checkpoint state for window {checkpoint_current_window}"
                )
        else:
            tplr.logger.info(
                f"No debug dict available for window {checkpoint_current_window}, skipping verification"
            )

    prev_param_state: dict[str, torch.Tensor] = {}
    param_avg_change: dict[str, torch.Tensor] = {}
    alpha: float = 0.20
    slice_idx = slice(0, 2)

    while start_w < target_w:
        tplr.logger.info(f"  • window {start_w}")

        # ------------------------------------------------------------------
        # 1) Fetch the aggregated object dumped by the leader validator.
        # ------------------------------------------------------------------
        if instance.is_master:
            # Clear memory before fetching to maximize available space
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fetch = await instance.comms.get(
                uid=str(leader_uid),
                window=start_w,
                key="aggregator",
                timeout=60,
                local=False,
                stale_retention=10,
                map_location=catchup_device,
            )

            # ── A. aggregated object exists → normal path ────────────────────
            if fetch.success and fetch.data is not None and "state_dict" in fetch.data:
                payload = fetch.data

                # ------------------------------------------------------------------
                # Re‑create the SimpleNamespace expected by `outer_step`.
                # ------------------------------------------------------------------
                gather_ns = SimpleNamespace(
                    state_dict=SimpleNamespace(**payload["state_dict"]),
                    uids=payload.get("uids", []),
                    skipped_uids=payload.get("skipped_uids", []),
                    success_rate=payload.get("success_rate", 0.0),
                )

                # Clear the original payload dict to free memory immediately
                del payload
                if hasattr(fetch, "data"):
                    fetch.data = None
                del fetch

            # ── B. aggregated object *missing* or *malformed* ────────────────
            else:
                gather_ns = None
                is_last_window = start_w == target_w - 1
                tplr.logger.warning(
                    "    ↳ %s – %s",
                    "not available" if fetch is None else "malformed payload",
                    "attempting gather‑fallback" if is_last_window else "skipping",
                )

                if is_last_window:
                    sync_block = (start_w + 1) * instance.hparams.blocks_per_window
                    ts_value = await instance.loop.run_in_executor(
                        None, instance.query_block_timestamp, sync_block
                    )
                    if ts_value is None:
                        tplr.logger.warning(
                            f"Could not get timestamp for sync block {sync_block}.",
                        )
                        time_min = time_max = None
                    else:
                        time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                        time_max = time_min + timedelta(
                            seconds=instance.hparams.time_window_delta_seconds
                        )

                    # ---- Gather fallback ----------------------------------------
                    gather_ns = await instance.comms.gather(
                        my_uid=instance.uid,
                        uids=instance.comms.peers,
                        window=start_w,
                        key="gradient",
                        timeout=45,
                        device=str(catchup_device),
                        local=False,
                        stale_retention=10,
                        totalks=instance.totalks,
                        compressor=instance.compressor,
                        time_min=time_min,
                        time_max=time_max,
                    )

                if gather_ns is None:
                    tplr.logger.warning("    ↳ gather‑fallback failed – skipping")
                else:
                    tplr.logger.info("    ↳ gather‑fallback succeeded – applying")
        else:
            gather_ns = None

        # Broadcast whether we should skip this window (master decides)
        if instance.is_master:
            skip_tensor = torch.tensor(
                [1 if gather_ns is None else 0],
                dtype=torch.int32,
                device=instance.config.device,
            )
        else:
            skip_tensor = torch.tensor(
                [0], dtype=torch.int32, device=instance.config.device
            )

        dist_helper.broadcast(skip_tensor, src=0)
        skip_window = bool(skip_tensor.item())

        # If skipping, continue to next window without updating scheduler
        if skip_window:
            # Don't increment global_step as no outer step was performed
            start_w += 1
            continue

        # ------------------------------------------------------------------
        # 2) All ranks apply the update.
        # ------------------------------------------------------------------
        # Synchronize all ranks before applying the outer step to ensure
        # they're processing the same window together
        dist_helper.safe_barrier("catchup_pre_outer_step", instance.local_rank)

        outer_step(
            instance.model,
            instance.outer_optimizer,
            gather_result=gather_ns,
            transformer=instance.transformer,
            compressor=instance.compressor,
            xshapes=instance.xshapes,
            totalks=instance.totalks,
            device=instance.config.device,
            is_master=instance.is_master,  # rank-0 handles logging
            world_size=instance.world_size,
            use_dct=instance.hparams.use_dct,
            wandb_run=instance.wandb
            if instance.is_master and isinstance(instance.wandb, Run)
            else None,
            global_step=instance.global_step,
        )

        # advance LR scheduler if one exists.
        inner_sched: LRScheduler | None = getattr(instance, "inner_scheduler", None)
        if inner_sched is not None:
            for _ in range(instance.hparams.inner_steps):
                # Respect flatten window during catch-up
                if not instance.should_skip_scheduler_step():
                    inner_sched.step()
                instance.inner_scheduler_step_count += 1

        # Aggressive memory cleanup after each window
        if instance.is_master and "gather_ns" in locals() and gather_ns is not None:
            # Clear the gather result to free memory
            if hasattr(gather_ns, "state_dict"):
                # Clear all attributes from state_dict namespace
                for key in list(vars(gather_ns.state_dict).keys()):
                    delattr(gather_ns.state_dict, key)
            del gather_ns

        # Force garbage collection to free CPU memory
        gc.collect()

        # Clear CUDA cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Log memory usage after cleanup
        if instance.is_master and (start_w - checkpoint_current_window) % 5 == 0:
            log_memory_usage(f"After window {start_w} cleanup")
        # ──────────────────────────────────────────────────────────────────────
        # 3) Debug‑dict comparison to estimate "how many steps behind" we are
        # ──────────────────────────────────────────────────────────────────────
        try:
            if instance.is_master:
                debug_fetch = await instance.comms.get(
                    uid=str(leader_uid),
                    window=start_w,
                    key="debug",
                    local=False,
                    stale_retention=10,
                )

                if debug_fetch.success and isinstance(debug_fetch.data, dict):
                    debug_dict = debug_fetch.data  # validator's payload

                    # --- update EMA of parameter‑slice changes ------------------
                    for name, p in instance.model.named_parameters():
                        if p.numel() < 2:
                            continue

                        # Handle DTensor parameters
                        if isinstance(p, DT):
                            curr_slice = (
                                p.to_local().detach().cpu().flatten()[slice_idx]
                            )
                        else:
                            curr_slice = p.detach().cpu().flatten()[slice_idx]

                        if name in prev_param_state:
                            delta = (curr_slice - prev_param_state[name]).abs()
                            if name not in param_avg_change:
                                param_avg_change[name] = delta.clone()
                            else:
                                param_avg_change[name].mul_(1 - alpha).add_(
                                    delta * alpha
                                )
                        prev_param_state[name] = curr_slice.clone()

                    # --- call shared comparison helper --------------------------
                    lr = instance.outer_optimizer.param_groups[0]["lr"]
                    cmp = await compare_model_with_debug_dict(
                        model=instance.model,
                        debug_dict=debug_dict,
                        learning_rate=lr,
                        index_range=(0, 2),
                        param_avg_change=param_avg_change,
                    )

                    if cmp["success"]:
                        tplr.logger.info(
                            f"[catch‑up] window {start_w} "
                            f"avg_steps_behind={cmp['avg_steps_behind']:.3f}, "
                            f"l2_norm={cmp['l2_norm']:.4f}"
                        )
                    else:
                        tplr.logger.warning(
                            f"[catch‑up] debug‑dict comparison failed for window {start_w}"
                        )
                else:
                    tplr.logger.warning(
                        f"[catch‑up] no debug‑dict found for window {start_w}"
                    )
        except Exception as exc:
            tplr.logger.warning(f"[catch‑up] debug‑dict processing error: {exc}")

        # Increment global_step since we performed an outer step
        instance.global_step += 1
        start_w += 1

        dist_helper.safe_barrier("catchup_post_window", instance.local_rank)

        # If the chain progressed while we were busy, extend the target.
        if instance.current_window > target_w:
            target_w = instance.current_window

    # Final aggressive memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Log final memory state
    log_memory_usage("Final memory state after catchup")
    tplr.logger.info("Catch‑up finished – model now in sync.")


async def compare_model_with_debug_dict(
    model: nn.Module,
    debug_dict: dict[str, list[float]],
    learning_rate: float,
    param_avg_change: dict[str, torch.Tensor] | None = None,
    *,
    min_step_size: float = 1e-9,
    index_range: tuple[int, int] = (0, 2),
) -> dict[str, bool | float | int]:
    """
    Compare weights with published debug snippets and return sync metrics.
    """
    # Initialize metrics
    total_squared_diff = 0.0
    total_abs_diff = 0.0
    param_count = 0
    max_diff = 0.0  # largest raw parameter diff
    max_steps = 0.0

    # Collect per‑tensor step‑ratio vectors so we can take
    # a single global median later
    tensors = 0
    step_ratio_list: list[torch.Tensor] = []

    named_params = (
        model.module.named_parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.named_parameters()
    )

    for name, p in named_params:
        key = name + "_debug"
        if key not in debug_dict or not isinstance(debug_dict[key], list):
            continue

        # --- grab the slice we care about --------------------------------
        if isinstance(p, DT):
            curr_slice = p.to_local().data.flatten()[index_range[0] : index_range[1]]
        else:
            curr_slice = p.data.flatten()[index_range[0] : index_range[1]]

        debug_slice = torch.tensor(
            debug_dict[key], dtype=p.dtype, device=curr_slice.device
        )

        diff_vec = curr_slice - debug_slice
        abs_vec = torch.abs(diff_vec)

        total_squared_diff += torch.sum(diff_vec**2).item()
        total_abs_diff += abs_vec.sum().item()
        raw_max = abs_vec.max().item()
        max_diff = max(max_diff, raw_max)
        param_count += abs_vec.numel()

        # --- element-wise steps-behind -----------------------------------
        if param_avg_change and name in param_avg_change:
            step_vec = torch.clamp(
                param_avg_change[name].to(curr_slice.device), min=min_step_size
            )
            if step_vec.numel() != abs_vec.numel():
                # fallback if stored slice has wrong length
                step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)
        else:
            step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)

        step_ratio = abs_vec / step_vec
        # Accumulate for global median
        step_ratio_list.append(step_ratio)
        max_steps = max(max_steps, step_ratio.max().item())
        tensors += 1

    l2_norm = math.sqrt(total_squared_diff)
    avg_l2_norm = math.inf if tensors == 0 else l2_norm / param_count
    avg_abs_diff = math.inf if tensors == 0 else total_abs_diff / param_count
    if not step_ratio_list:  # nothing compared
        median_steps = math.inf
        max_steps = math.inf
        interquartile_mean_steps = math.inf
    else:
        all_steps = torch.cat([t.flatten() for t in step_ratio_list])
        median_steps = all_steps.median().item()

        # Calculate interquartile mean (mean of values between Q1 and Q3)
        q1 = all_steps.quantile(0.25).item()
        q3 = all_steps.quantile(0.75).item()
        # Filter values in the interquartile range
        iqr_mask = (all_steps >= q1) & (all_steps <= q3)
        interquartile_mean_steps = all_steps[iqr_mask].mean().item()

    return {
        "success": True,
        "l2_norm": l2_norm,
        "avg_l2_norm": avg_l2_norm,
        "avg_abs_diff": avg_abs_diff,
        "max_diff": max_diff,
        "avg_steps_behind": interquartile_mean_steps,
        "interquartile_mean_steps_behind": interquartile_mean_steps,
        "max_steps_behind": max_steps,
        "param_count": param_count,
        "learning_rate": learning_rate,
    }


@torch.no_grad()
async def check_uid_index_overlap(
    neuron: NeuronT,
    gather_result: SimpleNamespace | list[SimpleNamespace],
    window: int,
    *,
    overlap_threshold: float = 0.90,
) -> dict:
    """
    For every peer-pair compute the per-chunk *set* overlap of their top-k index
    lists on each parameter.  A pair is flagged **only if the size-weighted
    average across *all* checked parameters** is ≥ `overlap_threshold`.

    Supports both merged results (typical usage) and partial results (for flexibility).
    """

    # ── 0. Extract UIDs and state_dicts from either partials or merged ────
    if isinstance(gather_result, list):
        # Partials: collect UIDs and state_dicts from all partials
        all_uids: list[int] = []
        all_state_dicts: list = []
        # Create UID-to-partial mapping for O(1) lookup (optimization for O(n×m) → O(n))
        uid_to_partial: dict[int, tuple[SimpleNamespace, int]] = {}

        for partial in gather_result:
            if partial is not None:
                partial_uids = getattr(partial, "uids", [])
                all_uids.extend(partial_uids)
                # Store state_dict along with its UIDs for later lookup
                for idx, uid in enumerate(partial_uids):
                    all_state_dicts.append(partial.state_dict)
                    # Map UID to (partial, index_in_partial) for O(1) lookup
                    uid_to_partial[uid] = (partial, idx)

        uids = all_uids
        # state_dicts[i] corresponds to uids[i]
    else:
        # Merged result: traditional path
        uids = list(getattr(gather_result, "uids", []))
        # All UIDs share the same merged state_dict
        all_state_dicts = [gather_result.state_dict] * len(uids)
        uid_to_partial = None  # Not needed for merged results
    Ptot = len(uids)
    if Ptot < 2:
        tplr.logger.info("[overlap] <2 peers – skip")
        return dict(
            pairs_checked=0,
            pairs_high_ovlap=0,
            ratio_high_ovlap=0.0,
            mean_overlap=0.0,
            min_overlap=0.0,
            max_overlap=0.0,
            pairs_over_thresh=[],
            uids_over_thresh={},
        )

    ts_map = dict(
        zip(
            uids,
            await asyncio.gather(
                *[neuron.comms.gradient_timestamp(uid, window - 1) for uid in uids]
            ),
        )
    )

    # ── 1. bookkeeping ────────────────────────────────────────────────────
    pair_acc: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    total_weighted_sum = 0.0
    total_weight = 0.0

    # ── 2. iterate over parameters that have compressed indices ───────────
    for pname, _ in neuron.model.named_parameters():
        idx_key = pname + "idxs"
        vals_key = pname + "vals"

        # Unpack all 12-bit packed indices using values shape
        # Now each UID has its own state_dict
        unpacked_indices = []
        valid_uids_for_param = []

        for i in range(Ptot):
            state_dict = all_state_dicts[i]

            # Get indices and values from this UID's state_dict
            if isinstance(gather_result, list):
                # Partials: use O(1) lookup instead of O(m) search
                # Direct lookup instead of searching (O(n×m) → O(n) optimization)
                if uids[i] not in uid_to_partial:
                    continue

                partial, uid_idx_in_partial = uid_to_partial[uids[i]]
                state_dict = partial.state_dict

                idxs_list = getattr(state_dict, idx_key, None)
                vals_list = getattr(state_dict, vals_key, None)

                if (
                    idxs_list is None
                    or vals_list is None
                    or uid_idx_in_partial >= len(idxs_list)
                ):
                    continue

                idx_data = idxs_list[uid_idx_in_partial]
                val_data = vals_list[uid_idx_in_partial]
            else:
                # Merged: traditional indexing
                idxs_list = getattr(state_dict, idx_key, None)
                vals_list = getattr(state_dict, vals_key, None)

                if idxs_list is None or vals_list is None:
                    continue

                idx_data = idxs_list[i] if isinstance(idxs_list, list) else idxs_list
                val_data = vals_list[i] if isinstance(vals_list, list) else vals_list

            # 12-bit packed format - use values shape for unpacking
            unpacked = unpack_12bit_indices(
                idx_data.to(neuron.config.device), val_data.shape
            )
            unpacked_indices.append(unpacked)
            valid_uids_for_param.append(i)

        if len(unpacked_indices) < 2:
            # Not enough peers have this parameter
            continue

        idxs_tensor = torch.stack(unpacked_indices, dim=0)
        P, *chunk_dims, k = idxs_tensor.shape
        C = int(torch.prod(torch.tensor(chunk_dims)))  # num chunks
        idxs_flat = idxs_tensor.reshape(P, C, k)

        param_weight = C * k  # size weight

        # Compare pairs using original UID indices for consistent keys
        for i in range(P):
            for j in range(i + 1, P):
                a = idxs_flat[i].unsqueeze(-1)  # (C,k,1)
                b = idxs_flat[j].unsqueeze(-2)  # (C,1,k)
                inter = (a == b).any(-1).sum(-1)  # (C,)
                mean_frac = (inter.float() / k).mean().item()

                total_weighted_sum += mean_frac * param_weight
                total_weight += param_weight

                # Use original UID indices (from valid_uids_for_param) for pair keys
                uid_idx_i = valid_uids_for_param[i]
                uid_idx_j = valid_uids_for_param[j]
                acc = pair_acc[(uid_idx_i, uid_idx_j)]
                acc[0] += mean_frac * param_weight
                acc[1] += param_weight

    # ── 3. second pass – decide offenders & track min/max ─────────────────
    pairs_high, pairs_over, uids_with_slashing = 0, [], {}
    min_pair, min_val = None, 1.0
    max_pair, max_val = None, 0.0

    for (i, j), (w_sum, w_tot) in pair_acc.items():
        avg_overlap = w_sum / w_tot if w_tot > 0 else 0.0

        # --- track global min / max --------------------------------------
        if avg_overlap < min_val:
            min_val, min_pair = avg_overlap, (uids[i], uids[j])
        if avg_overlap > max_val:
            max_val, max_pair = avg_overlap, (uids[i], uids[j])
        # ------------------------------------------------------------------

        if avg_overlap >= overlap_threshold:
            pairs_high += 1
            uid_i, uid_j = uids[i], uids[j]
            offender = uid_i if ts_map[uid_i] >= ts_map[uid_j] else uid_j
            uids_with_slashing[offender] = determine_slash_egregiousness(avg_overlap)

            pairs_over.append((uid_i, uid_j, avg_overlap))
            tplr.logger.debug(
                f"[overlap] peers {uid_i}/{uid_j} share "
                f"{avg_overlap * 100:.1f}% of indices (size-weighted avg)"
            )

    mean_overlap = total_weighted_sum / total_weight if total_weight else 0.0
    ratio_high = pairs_high / len(pair_acc) if pair_acc else 0.0

    # ── 4. summary log with min / max -------------------------------------
    tplr.logger.info(
        f"[overlap] {len(pair_acc)} pairs, {pairs_high} ≥{overlap_threshold * 100:.0f}% "
        f"({ratio_high * 100:.2f}%), size-weighted mean {mean_overlap * 100:.1f}%"
    )
    if min_pair is not None and max_pair is not None:
        tplr.logger.info(
            f"[overlap]   min {min_val * 100:.1f}%  (peers {min_pair[0]}/{min_pair[1]}) ; "
            f"max {max_val * 100:.1f}%  (peers {max_pair[0]}/{max_pair[1]})"
        )
    if uids_with_slashing:
        tplr.logger.warning(
            f"[overlap] offenders: {sorted(list(uids_with_slashing.keys()))}"
        )

    return dict(
        pairs_checked=len(pair_acc),
        pairs_high_ovlap=pairs_high,
        ratio_high_ovlap=ratio_high,
        mean_overlap=mean_overlap,
        min_overlap=min_val if min_pair is not None else 0.0,
        max_overlap=max_val if max_pair is not None else 0.0,
        pairs_over_thresh=pairs_over,
        uids_over_thresh=uids_with_slashing,
    )


def determine_slash_egregiousness(overlap_pct: float) -> str:
    """
    Based on the overlap_pct, return a level corresponding
    to an action which will be taken

    Args:
        overlap_pct: The percentage of overlap in the grads with
             other miners

    Returns:
        Category of overlap pct
    """

    invalid_number = overlap_pct < 0.0 or overlap_pct > 1.0
    if invalid_number:
        raise ValueError(f"overlap_pct must be between 0.0 and 1.0, got {overlap_pct}")

    egregiousness = "high"
    if overlap_pct >= 0.5:
        egregiousness = "max"
    if overlap_pct >= 0.6:
        egregiousness = "mega"

    return egregiousness


def instantiate_slashing_multiplier():
    """Centralize slashing config

    We multiply these percentages against the base final_score
    """
    return {
        "high": 0.5,  # case when similarity high
        "max": 0.0,  # case when similarity >= 95%
        "mega": 0.0,  # case when similarity = 100%
    }
