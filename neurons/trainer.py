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


# Standard library
import asyncio
import concurrent.futures
import os
import time
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.profiler as tp
from torch import autocast
from torch.distributed.tensor import DTensor as DT
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchtitan.components.loss import cross_entropy_loss

import tplr
from neurons.base_node import CPU_COUNT
from tplr import model_factory
from tplr.distributed import dist_helper
from tplr.muon import Muon, SingleDeviceMuonWithAuxAdam


class Trainer:
    """This will be an ongoing project to separate
    torch operations from comms/chain/miner/validator
    behavior
    """

    def __init__(self):
        pass

    def set_dataloader(self, validator: bool = False) -> None:
        self.dataset = self.dataset_manager.active_dataset

        shared_args = dict(
            dataset=self.dataset,
            uid=self.uid,
            window=self.current_window,
            steps_per_window=self.hparams.inner_steps,
            micro_bs=self.hparams.micro_batch_size,
            rank=self.rank,
            world_size=self.world_size,
            tp_degree=int(
                os.environ.get("TP_DEGREE", getattr(self.hparams, "tp_degree", 1))
            ),
        )

        if validator:
            SamplerClass = tplr.EvalSampler
            kwargs = shared_args | dict(
                batch_size=self.hparams.target_batch_size,
                validation_bs=self.hparams.validator_sample_micro_bs
                * self.hparams.micro_batch_size,
            )
        else:
            SamplerClass = tplr.MinerSampler
            kwargs = shared_args | dict(
                micro_bs=self.hparams.micro_batch_size,
                batch_size=self.hparams.batch_size,
                target_batch_size=self.hparams.target_batch_size,
            )

        self.sampler = SamplerClass(**kwargs)

        self.loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=self.hparams.micro_batch_size,
        )
        tplr.logger.info("[Run] dataset + sampler ready")
        return

    def get_expected_params(self) -> set[str]:
        """
        Creates a set of expected names for validation

            Returns: The names of all expected keys from a miner
        """
        expected_compressed_params = set()
        for param_name, _ in self.model.named_parameters():
            expected_compressed_params.add(param_name + "idxs")
            expected_compressed_params.add(param_name + "vals")
            expected_compressed_params.add(param_name + "quant_params")

        return expected_compressed_params

    def init_model(self, validator=False, meta=False):
        role = "miner"
        if validator:
            role = "validator"

        if meta:
            # Create model on meta device (no memory allocation)
            self.model = model_factory.create_meta_model(
                hparams=self.hparams,
                role=role,
                world_size=self.world_size,
            )
        else:
            # Initialize model with actual weights
            self.model = model_factory.initialize_torchtitan_model(
                hparams=self.hparams,
                role=role,
                device=str(self.device),
                world_size=self.world_size,
            )

        self.expected_compressed_params = self.get_expected_params()
        self.tokenizer = self.hparams.tokenizer

        return

    def init_optimizers_schedulers(self, validator=False):
        self.lr = float(self.hparams.outer_learning_rate)
        self.outer_optimizer = SGD(self.model.parameters(), lr=self.lr)

        self.inner_optimizer = self.get_inner_optimizer(validator)
        self.inner_scheduler = self.get_inner_scheduler()

        tplr.logger.info("[Init] optimizers & schedulers constructed")
        return

    def get_inner_scheduler(self):
        # Get optimizer configuration
        optimizer_config = getattr(self.hparams, "optimizer", {})
        optimizer_type = optimizer_config.get("type", "muon").lower()

        # Get optimizer-specific config (includes learning_rate and scheduler)
        opt_specific_config = optimizer_config.get(optimizer_type, {})
        scheduler_config = opt_specific_config.get("scheduler", {})

        # Get the effective learning rate used by the optimizer
        # Default based on optimizer type
        default_lr = 2e-4 if optimizer_type == "adamw" else 0.02
        effective_lr = opt_specific_config.get("learning_rate", default_lr)

        # Get scheduler parameters with optimizer-specific overrides
        warmup_steps = scheduler_config.get("warmup_steps", 750)
        t_max = scheduler_config.get("t_max", 20000)
        eta_min_factor = scheduler_config.get("eta_min_factor", 0.1)

        warmup_scheduler = lr_scheduler.LinearLR(
            self.inner_optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            self.inner_optimizer,
            T_max=t_max,
            eta_min=effective_lr * eta_min_factor,
        )
        inner_scheduler = lr_scheduler.SequentialLR(
            self.inner_optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        tplr.logger.info(
            f"[Init] Constructed {optimizer_type} scheduler with lr={effective_lr}, "
            f"warmup_steps={warmup_steps}, t_max={t_max}, eta_min_factor={eta_min_factor}"
        )

        return inner_scheduler

    def get_inner_optimizer(self, validator: bool):
        # Get optimizer config from hparams
        optimizer_config = getattr(self.hparams, "optimizer", {})
        optimizer_type = optimizer_config.get("type", "muon").lower()

        if validator:
            # inner scheduler and dummy optimizer for logging purposes
            _dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            # Any optimiser will do; SGD is the simplest and has no extra state.
            # Get the learning rate from the configured optimizer
            opt_specific_config = optimizer_config.get(optimizer_type, {})
            # Default based on optimizer type
            default_lr = 2e-4 if optimizer_type == "adamw" else 0.02
            lr = opt_specific_config.get("learning_rate", default_lr)
            inner_optimizer = torch.optim.SGD(
                [_dummy_param],
                lr=lr,
            )
        else:
            if optimizer_type == "adamw":
                adamw_config = optimizer_config.get("adamw", {})
                # Use optimizer-specific learning rate if provided
                adamw_lr = adamw_config.get("learning_rate", 2e-4)
                adamw_weight_decay = adamw_config.get("weight_decay", 0.1)
                inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=adamw_lr,
                    weight_decay=adamw_weight_decay,
                    betas=tuple(adamw_config.get("betas", [0.9, 0.95])),
                    eps=adamw_config.get("eps", 1e-8),
                    fused=True,
                    foreach=False,
                )
                tplr.logger.info(
                    f"[Init] Using AdamW inner optimizer with lr={adamw_lr}, "
                    f"weight_decay={adamw_weight_decay}, "
                    f"betas={adamw_config.get('betas', [0.9, 0.95])}"
                )
            elif optimizer_type == "muon":
                # Get bare model for parameter grouping

                # Get Muon-specific config
                muon_config = optimizer_config.get("muon", {})
                # Use optimizer-specific learning rate if provided
                muon_lr = muon_config.get("learning_rate", 0.02)

                # Check if we're using FSDP (DTensor parameters)
                is_fsdp = False
                for param in self.model.parameters():
                    if isinstance(param, DT):
                        is_fsdp = True
                        break

                # Separate parameters for Muon (2D matrices) and Adam (embeddings, scalars, head)
                hidden_2d_params = []
                embed_params = []
                scalar_params = []
                head_params = []

                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue

                    if (
                        param.ndim >= 2
                        and "embed" not in name
                        and "lm_head" not in name
                    ):
                        hidden_2d_params.append(param)
                    elif "embed" in name:
                        embed_params.append(param)
                    elif "lm_head" in name:
                        head_params.append(param)
                    else:
                        scalar_params.append(param)

                # Create parameter groups
                adam_groups = []
                if head_params:
                    adam_groups.append(
                        dict(
                            params=head_params,
                            lr=muon_lr * muon_config.get("head_lr_scale", 0.5),
                            weight_decay=self.hparams.weight_decay,
                        )
                    )
                if embed_params:
                    adam_groups.append(
                        dict(
                            params=embed_params,
                            lr=muon_lr * muon_config.get("embed_lr_scale", 0.5),
                            weight_decay=self.hparams.weight_decay,
                        )
                    )
                if scalar_params:
                    adam_groups.append(
                        dict(
                            params=scalar_params,
                            lr=muon_lr * muon_config.get("scalar_lr_scale", 0.2),
                            weight_decay=self.hparams.weight_decay,
                        )
                    )

                adam_groups = [
                    dict(**g, betas=(0.9, 0.95), eps=1e-8, use_muon=False)
                    for g in adam_groups
                ]

                if not hidden_2d_params:
                    tplr.logger.error(
                        "No hidden 2D parameters found for Muon optimizer"
                    )
                    raise ValueError("Model must have 2D weight matrices for Muon")

                # Use new Muon optimizer if FSDP is enabled, otherwise use single device version
                if is_fsdp:
                    # FSDP version with additional options
                    muon_group = dict(
                        params=hidden_2d_params,
                        lr=muon_lr,
                        momentum=muon_config.get("momentum", 0.95),
                        weight_decay=muon_config.get("weight_decay", 0.01),
                        use_muon=True,
                        rms_scale=muon_config.get(
                            "rms_scale", True
                        ),  # Add RMS scaling option
                        nesterov=muon_config.get(
                            "nesterov", True
                        ),  # Add Nesterov option
                    )
                    param_groups = adam_groups + [muon_group]
                    inner_optimizer = Muon(param_groups)
                    optimizer_name = "Muon (FSDP2)"
                else:
                    # Single device version with original options
                    muon_group = dict(
                        params=hidden_2d_params,
                        lr=muon_lr,
                        momentum=muon_config.get("momentum", 0.95),
                        weight_decay=muon_config.get("weight_decay", 0.01),
                        use_muon=True,
                    )
                    param_groups = adam_groups + [muon_group]
                    inner_optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
                    optimizer_name = "Muon (Single Device)"

                tplr.logger.info(
                    f"[Init] Using {optimizer_name} inner optimizer with lr={muon_lr}, "
                    f"momentum={muon_config.get('momentum', 0.95)}, weight_decay={muon_config.get('weight_decay', 0.01)}"
                )
                tplr.logger.info(
                    f"  - Hidden matrix params: {len(hidden_2d_params)} (Muon)"
                )
                tplr.logger.info(f"  - Embedding params: {len(embed_params)} (Adam)")
                tplr.logger.info(f"  - Scalar params: {len(scalar_params)} (Adam)")
                tplr.logger.info(f"  - Head params: {len(head_params)} (Adam)")
            else:
                raise ValueError(
                    f"Unknown optimizer type: {optimizer_type}. Supported: 'adamw', 'muon'"
                )

        return inner_optimizer

    async def evaluate_model(
        self,
        model: torch.nn.Module,
        loader: Iterable[torch.Tensor],
    ) -> tuple[float, int]:
        device: torch.device = next(model.parameters()).device
        total_loss = 0.0
        n_batches = 0

        world_size = dist_helper.world_size

        with torch.inference_mode():
            model.eval()
            for i, batch in enumerate(loader):
                if batch is None or len(batch) == 0:
                    tplr.log_with_context(
                        level="warning",
                        message=f"Empty batch at index {i}, skipping",
                        sync_window=self.sync_window,
                        current_window=self.current_window,
                    )
                    continue

                if isinstance(batch, torch.Tensor):
                    input_ids = batch.to(device, dtype=torch.long, non_blocking=True)
                else:
                    input_ids = torch.tensor(batch, dtype=torch.long, device=device)

                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]  # shift left by one
                labels[:, -1] = self.tokenizer.pad_token_id
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )

                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(input_ids)

                loss = cross_entropy_loss(logits, labels)

                total_loss += loss.item()
                n_batches += 1
                del input_ids, labels, logits
                torch.cuda.empty_cache()

                await asyncio.sleep(0)

        # Average loss across all ranks, sum batches for distributed training
        if world_size > 1 and dist_helper.is_distributed():
            total_loss = dist_helper.ddp_reduce(total_loss, device=device)
            n_batches = int(dist_helper.ddp_reduce(n_batches, device=device))

        return total_loss, n_batches

    # ------------------------------------------------------------------
    # Optimizer-state offload/prefetch (CPU <-> GPU) for inner optimizer
    # ------------------------------------------------------------------
    def _iter_inner_opt_state_tensors(self):
        """Yield (state_dict, key, tensor) triples for all tensor states of self.inner_optimizer."""
        opt = getattr(self, "inner_optimizer", None)
        if opt is None:
            return
        state = opt.state
        # opt.state uses Param tensors as keys; values are dicts with tensors like 'exp_avg', 'exp_avg_sq', etc.
        for p, s in state.items():
            if not isinstance(s, dict):
                continue
            for k, v in list(s.items()):
                if torch.is_tensor(v):
                    yield s, k, v

    def compute_adam_metrics(self) -> dict:
        """
        Compute comprehensive metrics from Adam-based optimizers (AdamW or Muon with Adam auxiliaries).
        This should be called BEFORE gradients are cleared.

        Handles distributed training by computing local squared norms and aggregating across ranks.

        Returns:
            dict: Metrics including grad norms, update norms, param norms, ratios, and optimizer hyperparams
        """
        metrics = {}

        # Local squared norms (will be summed across ranks then sqrt)
        local_grad_norm_sq = 0.0
        local_param_norm_sq = 0.0
        local_update_norm_sq = 0.0
        local_exp_avg_norm_sq = 0.0
        local_exp_avg_sq_norm_sq = 0.0

        # For max values, we'll collect locally then reduce with MAX operation
        local_grad_norm_max = 0.0
        local_update_norm_max = 0.0

        # For ratios, we need to compute per-parameter then average
        local_update_to_param_ratios_sum = 0.0
        local_ratio_count = 0

        num_params_with_grad = 0

        # Get optimizer config
        optimizer_config = getattr(self.hparams, "optimizer", {})
        optimizer_type = optimizer_config.get("type", "muon").lower()

        # Extract hyperparameters based on optimizer type
        if optimizer_type == "adamw":
            opt_config = optimizer_config.get("adamw", {})
            metrics["adam/lr"] = opt_config.get("learning_rate", 2e-4)
            metrics["adam/beta1"] = opt_config.get("betas", [0.9, 0.95])[0]
            metrics["adam/beta2"] = opt_config.get("betas", [0.9, 0.95])[1]
            metrics["adam/eps"] = opt_config.get("eps", 1e-8)
            metrics["adam/weight_decay"] = opt_config.get("weight_decay", 0.1)
        elif optimizer_type == "muon":
            opt_config = optimizer_config.get("muon", {})
            # For Muon, we use the Adam auxiliary optimizer parameters
            metrics["adam/lr"] = opt_config.get("learning_rate", 0.02) * opt_config.get(
                "embed_lr_scale", 0.5
            )
            metrics["adam/beta1"] = 0.9  # Default Adam betas used in Muon
            metrics["adam/beta2"] = 0.95
            metrics["adam/eps"] = 1e-8
            metrics["adam/weight_decay"] = self.hparams.weight_decay

        # Iterate through model parameters and optimizer state
        for p in self.model.parameters():
            if not p.requires_grad:
                continue

            # Compute parameter norm squared
            local_param_norm_sq += (p.norm() ** 2).item()

            # Compute gradient norm if gradient exists
            if p.grad is not None:
                num_params_with_grad += 1
                grad_norm = p.grad.norm().item()
                local_grad_norm_sq += grad_norm**2
                local_grad_norm_max = max(local_grad_norm_max, grad_norm)

            # Extract optimizer state if available
            if p in self.inner_optimizer.state:
                state = self.inner_optimizer.state[p]

                # Get momentum/exp_avg for update norm calculation
                if "exp_avg" in state:
                    exp_avg = state["exp_avg"]
                    exp_avg_norm = exp_avg.norm().item()
                    local_exp_avg_norm_sq += exp_avg_norm**2

                    # Update norm (same as exp_avg for Adam)
                    local_update_norm_sq += exp_avg_norm**2
                    local_update_norm_max = max(local_update_norm_max, exp_avg_norm)

                    # Update to param ratio (per-parameter)
                    param_norm = p.norm().item()
                    if param_norm > 1e-8:
                        local_update_to_param_ratios_sum += exp_avg_norm / param_norm
                        local_ratio_count += 1

                # Get exp_avg_sq if available (for AdamW)
                if "exp_avg_sq" in state:
                    exp_avg_sq = state["exp_avg_sq"]
                    local_exp_avg_sq_norm_sq += (exp_avg_sq.norm() ** 2).item()

        # Aggregate across ranks if distributed
        if self.world_size > 1 and dist_helper.is_distributed():
            from torch.distributed import ReduceOp

            # Sum squared norms across ranks, then sqrt
            global_grad_norm_sq = dist_helper.ddp_reduce(
                local_grad_norm_sq, device=self.device
            )
            global_param_norm_sq = dist_helper.ddp_reduce(
                local_param_norm_sq, device=self.device
            )
            global_update_norm_sq = dist_helper.ddp_reduce(
                local_update_norm_sq, device=self.device
            )
            global_exp_avg_norm_sq = dist_helper.ddp_reduce(
                local_exp_avg_norm_sq, device=self.device
            )
            global_exp_avg_sq_norm_sq = dist_helper.ddp_reduce(
                local_exp_avg_sq_norm_sq, device=self.device
            )

            # Max values across ranks
            global_grad_norm_max = dist_helper.ddp_reduce(
                local_grad_norm_max, op=ReduceOp.MAX, device=self.device
            )
            global_update_norm_max = dist_helper.ddp_reduce(
                local_update_norm_max, op=ReduceOp.MAX, device=self.device
            )

            # For ratio: sum and count across ranks, then divide
            global_ratio_sum = dist_helper.ddp_reduce(
                local_update_to_param_ratios_sum, device=self.device
            )
            global_ratio_count = int(
                dist_helper.ddp_reduce(local_ratio_count, device=self.device)
            )

            # Total params with grad across ranks
            num_params_with_grad = int(
                dist_helper.ddp_reduce(num_params_with_grad, device=self.device)
            )
        else:
            global_grad_norm_sq = local_grad_norm_sq
            global_param_norm_sq = local_param_norm_sq
            global_update_norm_sq = local_update_norm_sq
            global_exp_avg_norm_sq = local_exp_avg_norm_sq
            global_exp_avg_sq_norm_sq = local_exp_avg_sq_norm_sq
            global_grad_norm_max = local_grad_norm_max
            global_update_norm_max = local_update_norm_max
            global_ratio_sum = local_update_to_param_ratios_sum
            global_ratio_count = local_ratio_count

        # Compute final metrics (sqrt of summed squared norms = global L2 norm)
        metrics["adam/grad_norm_mean"] = (
            (global_grad_norm_sq**0.5) if global_grad_norm_sq > 0 else 0.0
        )
        metrics["adam/grad_norm_max"] = global_grad_norm_max
        metrics["adam/update_norm_mean"] = (
            (global_update_norm_sq**0.5) if global_update_norm_sq > 0 else 0.0
        )
        metrics["adam/update_norm_max"] = global_update_norm_max
        metrics["adam/param_norm_mean"] = (
            (global_param_norm_sq**0.5) if global_param_norm_sq > 0 else 0.0
        )
        metrics["adam/update_to_param_ratio_mean"] = (
            global_ratio_sum / global_ratio_count if global_ratio_count > 0 else 0.0
        )

        # Optional moment buffer checks
        metrics["adam/exp_avg_norm_mean"] = (
            (global_exp_avg_norm_sq**0.5) if global_exp_avg_norm_sq > 0 else 0.0
        )
        metrics["adam/exp_avg_sq_norm_mean"] = (
            (global_exp_avg_sq_norm_sq**0.5) if global_exp_avg_sq_norm_sq > 0 else 0.0
        )

        # Sanity check metric
        metrics["num_params_with_grad"] = num_params_with_grad

        return metrics

    def offload_inner_optimizer_states(self, *, log: bool = True) -> None:
        """
        Move inner optimizer state tensors (e.g., AdamW exp_avg, exp_avg_sq) to pinned CPU memory.
        Safe to call after an inner step finishes and before gradient compression.
        """
        if not getattr(self.hparams, "offload_optimizer_states", True):
            return
        opt = getattr(self, "inner_optimizer", None)
        if opt is None or getattr(self, "_inner_opt_offloaded", False):
            return
        moved_bytes = 0
        t0 = time.time()
        # Ensure all GPU kernels are done before D2H copies
        if torch.cuda.is_available():
            torch.cuda.synchronize(getattr(self, "device", None))
        for s, k, v in self._iter_inner_opt_state_tensors():
            if v.device.type == "cpu":
                # Pin CPU tensors to speed up subsequent H2D copies (no extra alloc if already pinned)
                if not v.is_pinned():
                    s[k] = v.pin_memory()
                continue
            # Asynchronous D2H into pinned CPU memory
            dst = torch.empty_like(v, device="cpu", pin_memory=True)
            dst.copy_(v, non_blocking=True)
            s[k] = dst  # drop GPU reference; GC frees it
            moved_bytes += v.element_size() * v.numel()
        # Ensure D2H copies complete before we proceed
        if torch.cuda.is_available():
            torch.cuda.synchronize(getattr(self, "device", None))
            torch.cuda.empty_cache()
        self._inner_opt_offloaded = True
        if log:
            tplr.logger.info(
                "[OptOffload] inner optimizer → CPU pinned | moved ~"
                f"{moved_bytes / 1e6:.1f} MB in {time.time() - t0:.3f}s"
            )

    def prefetch_inner_optimizer_states(self, *, log: bool = True) -> None:
        """
        Move inner optimizer state tensors back to GPU (device) asynchronously.
        Call at the very start of inner_steps() before any optimizer.step().
        """
        if not getattr(self.hparams, "offload_optimizer_states", True):
            return
        opt = getattr(self, "inner_optimizer", None)
        if opt is None or not getattr(self, "_inner_opt_offloaded", False):
            return
        device = getattr(
            self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        moved_bytes = 0
        t0 = time.time()
        for s, k, v in self._iter_inner_opt_state_tensors():
            if v.device == device:
                continue
            # Asynchronous H2D copy
            dst = torch.empty_like(v, device=device)
            dst.copy_(v, non_blocking=True)
            s[k] = dst
            moved_bytes += v.element_size() * v.numel()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        self._inner_opt_offloaded = False
        if log:
            tplr.logger.info(
                f"[OptOffload] inner optimizer ← GPU {device} | moved ~{moved_bytes / 1e6:.1f} MB in {time.time() - t0:.3f}s"
            )

    async def inner_steps(
        self,
        loader: DataLoader,
        step_window: int,
        null_round: bool = False,
    ) -> dict:
        """
        One inner-loop optimisation pass that is gradient-accumulation aware and
        synchronised across distributed ranks.

        Returns
        -------
        dict
            Keys: total_loss (float), batch_count (int), batch_tokens (int)
        """

        # Ensure the event loop and executor are initialized, which is normally
        # done in the `run()` method. This makes the method safe for isolated testing.
        if not hasattr(self, "loop"):
            self.loop = asyncio.get_running_loop()
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
            self.loop.set_default_executor(self.executor)

        # --- Prefetch inner optimizer states back to GPU, if they were offloaded ---
        # Do this once per window, not per micro-batch.
        self.prefetch_inner_optimizer_states()

        # Initialize profiler if config is available (from BaseNode)
        prof_config = getattr(self, "_prof_config", None)
        prof = None
        prof_ctx = nullcontext()

        if prof_config is not None:
            # Create the profiler context for this inner_steps call
            prof = tp.profile(**prof_config)
            prof_ctx = prof
            self._prof = prof  # Store for access in other methods if needed

        self.inner_optimizer.zero_grad()

        total_loss: float = 0.0
        batch_count: int = 0
        batch_tokens: int = 0  # local counter
        accum_batch_size: int = 0
        window_entry_loss: float = 0.0
        global_tokens: int = 0  # after cross-rank reduction
        global_loss_sum: float = 0.0
        local_tokens_sum: int = 0  # local running totals
        local_loss_sum: float = 0.0

        inner_step_count: int = 0
        loader_iter = iter(loader)

        # Wrap the training loop with profiler context (does nothing if prof_ctx is nullcontext)
        with prof_ctx:
            while not self.stop_event.is_set():
                # ------------------------------------------------------------------ #
                # 1. Fetch a batch (or detect EOS) on *each* rank
                # ------------------------------------------------------------------ #
                with tp.record_function("Data Loading") if prof else nullcontext():
                    try:
                        batch = await self.loop.run_in_executor(None, next, loader_iter)
                        local_has_batch = True
                    except StopIteration:
                        local_has_batch = False
                        batch = None

                # Decide collectively whether we should continue
                if self.world_size > 1:
                    cont = dist_helper.should_continue(local_has_batch, self.device)
                    if not cont:
                        if self.is_master:
                            tplr.logger.info(
                                "Stopping batch loop: at least one rank exhausted."
                            )
                        break
                    if not local_has_batch:  # exhausted only on this rank
                        continue

                # ------------------------------------------------------------------ #
                # 2. Prepare inputs
                # ------------------------------------------------------------------ #
                with tp.record_function("Prepare Inputs") if prof else nullcontext():
                    if isinstance(batch, torch.Tensor):
                        input_ids = batch.to(
                            self.device, dtype=torch.long, non_blocking=True
                        )
                    else:
                        input_ids = torch.tensor(
                            batch, dtype=torch.long, device=self.device
                        )

                    local_bs = len(batch)  # type: ignore
                    accum_batch_size += local_bs

                    tokens_this_batch = input_ids.numel()
                    batch_tokens += tokens_this_batch
                    local_tokens_sum += tokens_this_batch

                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]  # ✓ shift by +1
                    labels[:, -1] = self.tokenizer.pad_token_id
                    labels = torch.where(
                        labels == self.tokenizer.pad_token_id, -100, labels
                    )

                # ------------------------------------------------------------------ #
                # 3. Forward + backward
                # ------------------------------------------------------------------ #
                with tp.record_function("Forward Pass") if prof else nullcontext():
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        outputs = self.model(input_ids, labels)

                    calculated_loss = cross_entropy_loss(outputs, labels)

                    loss = calculated_loss / self.sampler.grad_accum_steps
                    loss_item = calculated_loss.detach().item()

                # -------------------------------------------------------------- #
                # 3-a.  Back-prop with no_sync() on non-final micro-batches
                # -------------------------------------------------------------- #
                with tp.record_function("Backward Pass") if prof else nullcontext():
                    corrected_accum_steps = max(self.sampler.grad_accum_steps, 1)
                    final_micro_batch = (batch_count + 1) % corrected_accum_steps == 0
                    if (
                        hasattr(self.model, "no_sync")
                        and self.world_size > 1
                        and not final_micro_batch
                    ):
                        sync_ctx = self.model.no_sync()
                    else:
                        sync_ctx = nullcontext()
                    with sync_ctx:
                        self.scaler.scale(loss).backward()

                total_loss += loss_item
                local_loss_sum += loss_item  # defer collective

                batch_count += 1
                window_changed = self.current_window != step_window

                # ------------------------------------------------------------------ #
                # 4. Decide *together* whether to take an optimiser step
                # ------------------------------------------------------------------ #
                step_now = final_micro_batch or window_changed
                if self.world_size > 1:
                    # Use MAX to ensure if any rank needs to step, all step
                    from torch.distributed import ReduceOp

                    step_now = bool(
                        dist_helper.ddp_reduce(
                            int(step_now), op=ReduceOp.MAX, device=self.device
                        )
                    )

                if step_now:
                    with (
                        tp.record_function("Optimizer Step") if prof else nullcontext()
                    ):
                        # ── one collective for scalar stats per inner step ───────────
                        global_tokens_step = int(
                            dist_helper.ddp_reduce(local_tokens_sum, device=self.device)
                        )
                        global_loss_step = dist_helper.ddp_reduce(
                            local_loss_sum, device=self.device
                        )
                        global_tokens += global_tokens_step
                        global_loss_sum += global_loss_step

                        # mean loss of this accumulation step for logging
                        from torch.distributed import ReduceOp

                        log_loss = dist_helper.ddp_reduce(
                            loss_item, op=ReduceOp.AVG, device=self.device
                        )

                        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float("inf")
                        )

                        if not null_round:
                            # Unscale, clip, then step via GradScaler if using fp16
                            self.scaler.unscale_(self.inner_optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.inner_optimizer)
                            self.scaler.update()
                            self.inner_scheduler.step()
                        else:
                            # Spin-up: don't step optimizer/scheduler, just clear gradients
                            self.scaler.update()

                        self.inner_optimizer.zero_grad(set_to_none=True)

                    inner_step_count += 1

                    # reset local accumulators BEFORE the next micro-batch
                    local_tokens_sum = 0
                    local_loss_sum = 0

                    accum_batch_size = int(
                        dist_helper.ddp_reduce(accum_batch_size, device=self.device)
                    )
                    if self.is_master:
                        # Convert grad_norm to scalar (handles both Tensor and DTensor)
                        grad_norm_scalar = float(
                            grad_norm_before_clip.item()
                            if hasattr(grad_norm_before_clip, "item")
                            else grad_norm_before_clip
                        )

                        tplr.logger.info(
                            f"Inner Step {inner_step_count}, "
                            f"Batch {batch_count}, loss: {log_loss:.4f}, "
                            f"grad_norm: {grad_norm_scalar:.4f}, "
                            f"accum: {accum_batch_size}/{self.hparams.batch_size}"
                        )
                    if window_entry_loss == 0.0:
                        total_batches_first_step = int(
                            dist_helper.ddp_reduce(batch_count, device=self.device)
                        )
                        window_entry_loss = global_loss_sum / total_batches_first_step
                    accum_batch_size = 0  # reset on *all* ranks

                    # Step the profiler after optimizer step
                    if prof is not None and hasattr(prof, "step"):
                        prof.step()

                # ------------------------------------------------------------------ #
                # 5. Outer-loop window control
                # ------------------------------------------------------------------ #
                need_sync = (
                    window_changed or inner_step_count == self.hparams.inner_steps
                )
                if self.world_size > 1:
                    from torch.distributed import ReduceOp

                    global_done = bool(
                        dist_helper.ddp_reduce(
                            int(need_sync), op=ReduceOp.MAX, device=self.device
                        )
                    )
                else:
                    global_done = need_sync

                if global_done:
                    if self.is_master:
                        tplr.logger.info("<Exhausted window: exiting synchronously>")
                    if not null_round:
                        for _ in range(inner_step_count, self.hparams.inner_steps):
                            self.inner_scheduler.step()
                    break

            await asyncio.sleep(0)

        # ---------------------------------------------------------------------- #
        # 6. Compute gradient and optimizer metrics BEFORE clearing gradients
        # ---------------------------------------------------------------------- #
        adam_metrics = self.compute_adam_metrics()

        # Compute global gradient and weight norms (squared sum across ranks, then stats)
        local_grad_norms_sq = []
        local_weight_norms_sq = []

        for p in self.model.parameters():
            if p.requires_grad:
                local_weight_norms_sq.append((p.norm() ** 2).item())
                if p.grad is not None:
                    local_grad_norms_sq.append((p.grad.norm() ** 2).item())

        # Sum squared norms across ranks
        total_grad_norm_sq = sum(local_grad_norms_sq) if local_grad_norms_sq else 0.0
        total_weight_norm_sq = (
            sum(local_weight_norms_sq) if local_weight_norms_sq else 0.0
        )

        if self.world_size > 1 and dist_helper.is_distributed():
            total_grad_norm_sq = dist_helper.ddp_reduce(
                total_grad_norm_sq, device=self.device
            )
            total_weight_norm_sq = dist_helper.ddp_reduce(
                total_weight_norm_sq, device=self.device
            )

        # Global norms (sqrt of summed squares)
        global_grad_norm = (total_grad_norm_sq**0.5) if total_grad_norm_sq > 0 else 0.0
        global_weight_norm = (
            (total_weight_norm_sq**0.5) if total_weight_norm_sq > 0 else 0.0
        )

        # ---------------------------------------------------------------------- #
        # 7. Return aggregated metrics
        # ---------------------------------------------------------------------- #
        batch_count = int(dist_helper.ddp_reduce(batch_count, device=self.device))
        return {
            "total_loss": global_loss_sum,  # cross-rank sum
            "window_entry_loss": window_entry_loss,
            "batch_count": batch_count,  # cross-rank sum
            "batch_tokens": global_tokens,  # cross-rank sum
            "global_grad_norm": global_grad_norm,  # global gradient norm
            "global_weight_norm": global_weight_norm,  # global weight norm
            "adam_metrics": adam_metrics,  # Adam optimizer metrics dict
        }

    def outer_step(self, gather_result, log_wandb: bool = False):
        return tplr.neurons.outer_step(
            self.model,
            self.outer_optimizer,
            gather_result=gather_result,
            transformer=self.transformer,
            compressor=self.compressor,
            xshapes=self.xshapes,
            totalks=self.totalks,
            device=str(self.device),
            is_master=self.is_master,
            world_size=self.world_size,
            use_dct=self.hparams.use_dct,
            wandb_run=self.wandb if self.is_master and log_wandb else None,
            global_step=self.global_step,
        )
