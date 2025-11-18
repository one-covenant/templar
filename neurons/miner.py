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
import argparse
import asyncio
import concurrent.futures
import gc
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

# Set PyTorch CUDA memory allocator to use expandable segments for better FSDP performance
# This reduces memory fragmentation and improves allocation efficiency for large models
# Must be set before importing torch
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from types import SimpleNamespace
from typing import cast

import bittensor as bt
import numpy as np
import torch
import uvloop
from torch.amp.grad_scaler import GradScaler
from torch.distributed.tensor import DTensor as DT

import tplr
from neurons import BaseNode, Trainer
from neurons.base_node import CPU_COUNT
from tplr import model_factory
from tplr.distributed import dist_helper

# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Miner(BaseNode, Trainer):
    def log_gpu_memory(self, stage: str):
        """Log current GPU memory allocation and reservation"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            tplr.logger.info(
                f"[GPU Memory - {stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )

    def check_memory_threshold(self, threshold_gb: float = 0.5):
        """Check if available memory is below threshold and cleanup if needed"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            max_memory = (
                torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            )
            available = max_memory - allocated

            if available < threshold_gb:
                tplr.logger.warning(f"Low GPU memory: {available:.2f} GB available")
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
                self.log_gpu_memory("After emergency cleanup")

    # Command line config items.
    @staticmethod
    def miner_config():
        parser = argparse.ArgumentParser(description="Miner script")
        parser.add_argument(
            "--netuid", type=int, default=268, help="Bittensor network UID."
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project."
        )
        parser.add_argument(
            "--actual-batch-size",
            type=int,
            default=None,
            help="Override the batch size defined in hparams.",
        )
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument(
            "--amp-dtype",
            choices=["bf16", "fp16"],
            default="bf16",
            help="Mixed-precision data type. Use «fp16» to enable GradScaler.",
        )
        parser.add_argument(
            "--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0))
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument(
            "--store-gathers",
            action="store_true",
            help="Store gathered gradients in R2",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode - use all peers without filtering",
        )
        parser.add_argument(
            "--local",
            action="store_true",
            help="Local run - use toy model, small enough for a laptop.",
        )
        parser.add_argument(
            "--profile-iters",
            type=int,
            default=0,
            help="Active iterations per Torch‑Profiler trace (0 = disable)",
        )
        parser.add_argument(
            "--profile-dir",
            type=str,
            default="./log/profiler",
            help="Directory to save profiler traces",
        )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()

        return config

    def __init__(self):
        tplr.logger.debug("Starting initialization...")

        # Init config and load hparams
        self.config = Miner.miner_config()
        # ---------------------------------------------------------------------
        # Distributed initialisation
        # ---------------------------------------------------------------------
        dist_helper.init_process_group(backend="nccl", timeout_minutes=45)
        self.rank = dist_helper.rank
        self.world_size = dist_helper.world_size
        self.local_rank = dist_helper.local_rank
        self.is_master = dist_helper.is_master

        if dist_helper.device:
            self.device = dist_helper.device
            self.config.device = str(dist_helper.device)
        else:
            self.config.device = self.config.device or "cuda"
            self.device = torch.device(self.config.device)
        tplr.logger.info(f"[Init] device set → {self.device}")

        # Mixed precision setup
        self.amp_dtype = (
            torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        )
        self.scaler = GradScaler(
            enabled=(self.amp_dtype is torch.float16 and self.device.type == "cuda")
        )
        tplr.logger.info(
            f"[Init] Using {self.config.amp_dtype}. GradScaler enabled: {self.scaler.is_enabled()}"
        )

        # Convenience flags - already set from dist_helper
        self.config.local = cast(bool, self.config.local)
        self.hparams = tplr.load_hparams(use_local_run_hparams=self.config.local)

        if self.config.actual_batch_size is not None:
            tplr.logger.info(
                f"Overriding hparams batch size: {self.hparams.batch_size} -> {self.config.actual_batch_size}"
            )
            self.hparams.batch_size = self.config.actual_batch_size

        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        tplr.logger.info("[Init] Bittensor wallet loaded")
        super().__init__()

        # Initialize model on meta device first
        self.init_model(meta=True)
        # Move model from meta to actual device (allocates memory but no initialization)
        self.model = self.model.to_empty(device=str(self.device))
        self.model_initialized = False  # Track if model has actual weights

        # Store parallelization parameters for later use
        tt = getattr(self.hparams, "torchtitan", SimpleNamespace())
        # Check environment override first, parse safely, clamp to ≥1
        raw_tp = os.environ.get("TP_DEGREE")
        if raw_tp is not None:
            try:
                self.tp_degree = max(1, int(raw_tp))
            except (ValueError, TypeError):
                tplr.logger.warning(
                    f"Invalid TP_DEGREE='{raw_tp}', falling back to hparams"
                )
                self.tp_degree = int(getattr(tt, "tp_degree", 1))
        else:
            self.tp_degree = int(getattr(tt, "tp_degree", 1))
        self.pp_degree = int(getattr(tt, "pp_degree", 1))
        self.cp_degree = int(getattr(tt, "cp_degree", 1))
        self.dp_replicate = int(getattr(tt, "dp_replicate", 1))
        self.dp_shard = int(getattr(tt, "dp_shard", 1))

        # Init compression
        self.transformer = tplr.compress.ChunkingTransformer(
            self.model, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.TopKCompressor(
            use_quantization=True,
            quantization_bins=self.hparams.quantization_bins,
            quantization_range=self.hparams.quantization_range,
        )
        tplr.logger.info("[Init] compression pipeline ready")

        # Init optimizer and momentum
        self.init_optimizers_schedulers()

        self.error_feedback = {}
        self.error_feedback_cpu_buffers = {}
        self.owned_params = set()

        self.xshapes = {}
        self.totalks = {}
        model_iterator = self.model.named_parameters()

        # [TP] In TP, all ranks in the same TP group must own the SAME parameters
        # Use already-validated self.tp_degree instead of re-reading environment
        if (
            self.tp_degree > 1
            and self.world_size >= self.tp_degree
            and (self.world_size % self.tp_degree == 0)
        ):
            # Calculate DP rank (which TP group this rank belongs to)
            dp_rank = self.rank // self.tp_degree
            dp_world_size = self.world_size // self.tp_degree
        else:
            dp_rank = self.rank
            dp_world_size = self.world_size

        # Track error feedback buffer memory
        total_buffer_memory = 0
        num_dtensor_params = 0
        num_regular_params = 0

        for idx, (n, p) in enumerate(model_iterator):
            # Guard against ZeroDivisionError in edge cases
            if dp_world_size > 0 and idx % dp_world_size == dp_rank:
                # this rank "owns" the parameter
                self.owned_params.add(n)
                # For DTensors, create error feedback based on local shard size
                self.error_feedback[n] = None
                # Use local shape for DTensor to avoid allocating full-sized buffers
                if isinstance(p, DT):
                    local = p.to_local()
                    local_shape = local.shape
                    self.error_feedback_cpu_buffers[n] = torch.empty(
                        local_shape, device="cpu", pin_memory=True
                    )
                    buffer_size = local.numel() * local.element_size()
                    total_buffer_memory += buffer_size
                    num_dtensor_params += 1
                else:
                    self.error_feedback_cpu_buffers[n] = torch.empty(
                        p.shape, device="cpu", pin_memory=True
                    )
                    buffer_size = p.numel() * p.element_size()
                    total_buffer_memory += buffer_size
                    num_regular_params += 1

            enc = self.transformer.encode(
                torch.empty(p.shape, dtype=torch.float16, device=self.device),
                use_dct=self.hparams.use_dct,
            )
            _, _, xshape, totalk, _ = self.compressor.compress(
                enc,
                self.hparams.topk_compression,
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

        # Log error feedback buffer memory allocation
        total_buffer_gb = total_buffer_memory / (1024**3)
        total_owned_params = num_dtensor_params + num_regular_params
        tplr.logger.info(
            f"[Init] Error feedback buffers allocated: {total_buffer_gb:.2f} GB "
            f"({total_owned_params} owned params: {num_dtensor_params} DTensor, {num_regular_params} regular) "
            f"[TP={self.tp_degree}, DP_rank={dp_rank}/{dp_world_size}]"
        )

        tplr.logger.info(
            f"[Init] Compression initialized for {len(self.xshapes)} parameters"
        )

        self.bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)
        tplr.logger.info(
            f"[Miner] code_version={tplr.__version__} "
            f"checkpoint_init_flag={self.bootstrap_version or '<none>'}"
        )

        # Calculate the number of warmup windows before the first real training step
        self.warmup_windows = (
            self.hparams.validator_offset + self.hparams.peer_list_window_margin
        )
        tplr.logger.info(
            f"[Init] Warmup windows before first real training: {self.warmup_windows}"
        )

        # Init comms
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            hparams=self.hparams,
            uid=None,  # UID will be set after comms is initialized
        )

        if self.wallet.hotkey.ss58_address not in self.comms.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.comms.metagraph.netuid}[/bold]"
            )
            sys.exit()
        self.uid = self.comms.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.comms.uid = self.uid

        self.ckpt = tplr.DCPCheckpointer(
            self.comms, uid=self.uid, version=tplr.__version__, repo_root="."
        )

        self.bucket = self.comms.get_own_bucket("gradients", "read")
        if self.is_master:
            self.comms.try_commit(self.wallet, self.bucket)
        dist_helper.safe_barrier("post_init", self.local_rank)

        # Init state params
        self.current_block = self.comms.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        tplr.logger.info(
            f"[Init] chain at block {self.current_block}, window {self.current_window}"
        )

        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window
        self.step_counter = 0

        # Track additional metrics
        self.total_tokens_processed = 0

        if self.is_master:
            # Initialize WandB
            self.wandb = tplr.initialize_wandb(
                run_prefix="M",
                uid=self.uid,
                config=self.config,
                group="miner",
                job_type="mining",
            )
            tplr.logger.info("[Init] WandB session started")

            # Initialize metrics logger for InfluxDB
            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="M",
                uid=self.uid,
                config=self.config,
                role="miner",
                group="miner",
                job_type="mining",
            )

        # Initialize peer related attributes
        self.next_peers: list[int] | None = None
        self.next_reserve_peers: list[int] | None = None
        self.peers_update_window = -1

        self.dataset_manager = tplr.sharded_dataset.ShardedDatasetManager(
            sequence_length=self.hparams.sequence_length,
            rank=self.local_rank,
            world_size=self.world_size,
            comms=self.comms,
            token_dtype=np.uint32,  # Match preprocessing script dtype
        )
        self.outer_steps_per_shard = getattr(self.hparams, "outer_steps_per_shard")

        tplr.logger.info("[Init] ✔ fully done – entering run()")

    # Main training loop.
    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        # Use config peers if provided
        if self.config.peers:
            self.comms.peers = self.config.peers

        self.comms.commitments = await self.comms.get_commitments()
        tplr.logger.info("Loaded commitments")

        peer_start = tplr.T()
        # Fetch peers and get start_window from highest stake validator
        if self.is_master:
            await tplr.neurons.update_peers(
                instance=self, window=self.current_window, peer_start=peer_start
            )

            start_window = await self.comms.get_start_window()
            tplr.logger.info(f"Using start_window: {start_window}")

            val = -1 if start_window is None else start_window
            tensor = torch.tensor([val], dtype=torch.long, device=self.device)
        else:
            tensor = torch.zeros(1, dtype=torch.long, device=self.device)

        dist_helper.broadcast(tensor, src=0)
        val = tensor.item()
        start_window = None if val == -1 else int(val)
        assert start_window is not None
        self.start_window = start_window

        # global_step tracks actual outer steps performed (starts at 0)
        self.global_step = 0
        # ------------------------------------------------------------------
        # Proceed to load checkpoint using consolidated logic
        #   • Check if current version checkpoint exists
        #   • If not, fall back to bootstrap version if configured
        #   • rank-0 (or single-GPU run) downloads & catches-up
        #   • remaining ranks receive state via NCCL broadcast
        # ------------------------------------------------------------------

        # Use consolidated checkpoint loading function
        (
            ckpt_ok,
            ckpt_sync_win,
            ckpt_global_step,
            from_bootstrap,
        ) = await tplr.neurons.load_checkpoint_with_fallback(self)

        # If no checkpoint was loaded, initialize model weights now
        if not self.model_initialized:
            tplr.logger.info("No checkpoint loaded, initializing model weights...")
            # Initialize weights in-place on the existing model
            model_factory.initialize_weights_inplace(self.model, self.hparams)
            self.model_initialized = True

        # Handle catch-up and scheduler replay using consolidated logic
        await tplr.neurons.handle_checkpoint_catchup(
            self, ckpt_ok, ckpt_sync_win, ckpt_global_step, from_bootstrap
        )

        self.comms.start_commitment_fetcher()

        current_shard = self.global_step // self.outer_steps_per_shard
        tplr.logger.info(
            f"Starting with global_step={self.global_step} (actual outer steps)"
        )

        # Initialize datasets (only rank 0 downloads, handled internally by dataset_manager)
        _ = await self.dataset_manager.initialize_datasets(current_shard)

        # Synchronize all ranks after dataset initialization
        dist_helper.safe_barrier("dataset_init_complete", self.local_rank)

        # All workers need to instantiate dataloader
        self.set_dataloader()

        # Track the current shard to avoid double-swapping at initialization
        last_shard = current_shard

        # Put a dummy gradient to mark this miner as active for validators
        if self.is_master:
            tplr.logger.info("Putting dummy gradient to mark miner as active...")
            dummy_gradient = {
                "metadata": {"window": self.current_window, "dummy": True}
            }
            await self.comms.put(
                state_dict=dummy_gradient,
                uid=str(self.uid),
                window=self.current_window,
                key="gradient",
                local=False,
            )
            tplr.logger.info("Dummy gradient posted successfully")

        while not self.stop_event.is_set():
            await asyncio.sleep(0)
            # 1. Initialize window and update peers
            window_start = tplr.T()
            # Start the gather in the background:
            step_window = self.current_window
            # global_step will be incremented only when we do an actual outer step
            tplr.logger.info(
                f"\n{'-' * 40} Window: {step_window} (Outer Steps Taken: {self.global_step}) {'-' * 40}"
            )

            peer_start = tplr.T()
            if self.is_master:
                await tplr.neurons.update_peers(
                    instance=self, window=step_window, peer_start=peer_start
                )
            peer_update_time = tplr.T() - peer_start

            # 2. Load data
            data_start = tplr.T()

            # Update sampler for current window
            self.sampler.set_window_uid(self.uid, step_window)

            # Check if we need to swap dataset based on shard index change
            current_shard_check = self.global_step // self.outer_steps_per_shard
            if current_shard_check > last_shard:
                tplr.logger.info(
                    f"Swapping dataset after {self.global_step} outer steps at window {step_window}"
                )
                await self.dataset_manager.swap_datasets()
                self.set_dataloader()
                dist_helper.safe_barrier("sync_shard_switch", self.local_rank)
                last_shard = current_shard_check

            data_loading_time = tplr.T() - data_start
            tplr.logger.info(
                f"{tplr.P(step_window, data_loading_time)} Loaded training data"
            )

            # Offload parameters to CPU before inner_steps
            offload_start = time.time()
            params_offloaded, param_specs = dist_helper.get_offloaded_params(self.model)
            offload_time = time.time() - offload_start
            tplr.logger.info(f"Parameter offload to CPU took {offload_time:.4f}s")

            # 3. Accumulate gradients over batches
            train_start = tplr.T()
            # Check if we're in a null round (warmup phase or no gather peers)
            window_offset = self.current_window - (
                self.start_window or self.current_window
            )
            warmup_null = window_offset < self.warmup_windows
            no_peers_null = len(self.comms.peers) == 0

            # Broadcast null round decision to all ranks - all ranks must agree to do null round
            null_round = dist_helper.all_agree(
                warmup_null or no_peers_null, self.device, "null_round_check"
            )

            if null_round:
                if warmup_null:
                    tplr.logger.info(
                        f"Start accumulating... (null round: warmup {window_offset + 1}/{self.warmup_windows})"
                    )
                elif no_peers_null:
                    tplr.logger.info(
                        "Start accumulating... (null round: no gather peers available)"
                    )
                else:
                    tplr.logger.info(
                        "Start accumulating... (null round: triggered by another rank)"
                    )
            else:
                tplr.logger.info("Start accumulating...")

            res = await self.inner_steps(
                loader=self.loader, step_window=step_window, null_round=null_round
            )

            # Restore parameters from CPU after inner_steps
            restore_start = time.time()
            dist_helper.restore_offloaded_params(
                self.model, params_offloaded, param_specs
            )
            restore_time = time.time() - restore_start
            tplr.logger.info(f"Parameter restore to GPU took {restore_time:.4f}s")

            training_time = tplr.T() - train_start
            window_entry_loss = res["window_entry_loss"]
            n_batches = res["batch_count"]
            window_tokens = res["batch_tokens"]
            global_grad_norm = res["global_grad_norm"]
            global_weight_norm = res["global_weight_norm"]
            adam_metrics = res["adam_metrics"]

            # Free VRAM pressure during compression by offloading inner opt states to CPU
            # (they are not needed until the next inner_steps call).
            try:
                self.offload_inner_optimizer_states()
            except Exception:
                tplr.logger.warning(
                    "Optimizer-state offload failed; continuing.", exc_info=True
                )

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # If training finishes early, wait until the *next* chain-window starts.

            tplr.logger.info(
                f"{tplr.P(step_window, tplr.T() - train_start)} Completed training"
            )

            # Synchronise all ranks
            dist_helper.safe_barrier("pre_gather", self.local_rank)

            # 1️⃣ every rank builds its momentum shard
            compress_start = tplr.T()
            self.log_gpu_memory("Before prepare_gradient_dict")
            torch.cuda.reset_peak_memory_stats(self.device)

            shard_gradient, _, _ = tplr.prepare_gradient_dict(
                self, step_window, null_round
            )

            peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
            self.log_gpu_memory("After prepare_gradient_dict")
            tplr.logger.info(f"[GPU] Peak during compression: {peak:.2f} GB")
            compression_time = tplr.T() - compress_start
            tplr.logger.info(
                f"{tplr.P(step_window, compression_time)} "
                f"Compressed local shard with {len(shard_gradient) - 1} tensors"
            )

            # gather the shards → rank-0
            gathered = dist_helper.gather_object(
                shard_gradient,
                object_list=[None] * self.world_size if self.is_master else None,
                dst=0,
            )

            # ------------------------------------------------------------
            #  rank-0 merges & uploads the full gradient
            # Handles three cases: Pure FSDP, Pure TP, and TP+FSDP hybrid
            # ------------------------------------------------------------
            gradient = {}
            processed_state_dict = {}  # Initialize here for proper scope
            tp_degree = self.tp_degree if self.tp_degree > 1 else 1

            if self.is_master:
                assert gathered is not None

                if tp_degree == 1:
                    # Case 1: Pure FSDP - each rank owns different parameters
                    # Simple merge with no conflicts
                    for i, shard in enumerate(gathered):
                        if shard:
                            gradient.update(shard)
                            gathered[i] = None  # Free memory immediately

                elif self.world_size == tp_degree:
                    # Case 2: Pure TP - all ranks have same parameters sharded
                    # Need to reconstruct full parameters from shards
                    gradient = self._reconstruct_tp_gradients(gathered)

                else:
                    # Case 3: TP + FSDP hybrid
                    # Process each FSDP group's TP shards separately
                    num_fsdp_groups = self.world_size // tp_degree

                    for fsdp_id in range(num_fsdp_groups):
                        # Extract TP shards for this FSDP group
                        start_idx = fsdp_id * tp_degree
                        end_idx = start_idx + tp_degree
                        tp_shards = gathered[start_idx:end_idx]

                        # Reconstruct this FSDP group's parameters
                        reconstructed = self._reconstruct_tp_gradients(tp_shards)
                        gradient.update(reconstructed)

                        # Free memory for processed shards
                        for idx in range(start_idx, end_idx):
                            gathered[idx] = None

                # dataset metadata
                gidx = self.sampler._global_indices()
                ids = self.sampler.ids_for_indices(gidx.tolist())
                h = hashlib.blake2b(digest_size=16)
                h.update(np.asarray(sorted(ids), dtype=np.uint64).tobytes())
                sample_digest = h.hexdigest()
                sample_count = len(ids)

                # ── attach window + sample receipt ─────────────────────
                # Build xshapes and totalks for TP-reconstructed parameters
                final_xshapes = {}
                final_totalks = {}

                for key in gradient:
                    if key.endswith("idxs"):
                        param_name = key[:-4]  # Remove "idxs" suffix

                        # Get shape and totalk from the reconstructed gradient
                        if param_name + "vals" in gradient:
                            idxs_tensor = gradient[key]
                            vals_tensor = gradient[param_name + "vals"]

                            # For TP-reconstructed params, use the full shape
                            shard_meta = gradient.get(param_name + "shard_metadata", {})
                            if shard_meta.get("was_tp_sharded"):
                                # Use the global shape from reconstruction
                                final_xshapes[param_name] = shard_meta.get(
                                    "global_shape", vals_tensor.shape
                                )
                                # Use the totalk from reconstruction
                                final_totalks[param_name] = shard_meta.get(
                                    "totalk", len(idxs_tensor)
                                )
                            else:
                                # Use the original shape and totalk
                                final_xshapes[param_name] = self.xshapes.get(
                                    param_name, vals_tensor.shape
                                )
                                final_totalks[param_name] = self.totalks.get(
                                    param_name, len(idxs_tensor)
                                )

                # Update metadata (preserve existing fields if any)
                if "metadata" not in gradient:
                    gradient["metadata"] = {}
                gradient["metadata"].update(
                    {
                        "window": step_window,
                        "sample_digest": sample_digest,
                        "sample_count": sample_count,
                        "xshapes": final_xshapes,  # Critical for validator decompression
                        "totalks": final_totalks,  # Critical for validator decompression
                    }
                )

                mode = (
                    "FSDP"
                    if tp_degree == 1
                    else ("TP" if self.world_size == tp_degree else "TP+FSDP")
                )
                tplr.logger.info(
                    f"Gradient ready: {len(gradient) - 1} tensors ({mode} mode)"
                )
                del gathered  # Free gathered list after merging

                # move to CPU before R2 upload
                processed_state_dict = {
                    k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
                    for k, v in gradient.items()
                }

            else:
                # non-master ranks simply wait; they don't upload
                put_time = 0.0
                if self.world_size > 1:
                    del gathered  # Free gathered list on non-master ranks too

            tplr.logger.info(f"Stopped accumulating: {n_batches} batches")
            dist_helper.safe_barrier("post_gather", self.local_rank)

            if self.current_window == step_window:
                tplr.logger.info(
                    "Training complete; waiting for window to be exhausted..."
                )
                await self.wait_until_window(step_window + 1)

            if self.is_master:
                # OOM TEST: Skip gradient upload
                tplr.logger.info("🔥 OOM TEST: Skipping gradient upload")
                put_time = 0.0

                # Free memory immediately
                del processed_state_dict
                del gradient
                torch.cuda.empty_cache()

            sync_block = self.current_window * self.hparams.blocks_per_window
            ts_value = await self.loop.run_in_executor(
                None, self.query_block_timestamp, sync_block
            )
            if ts_value is None:
                tplr.logger.warning(
                    f"Could not get timestamp for sync block {sync_block}. Using current time as fall back.",
                )
                ts_value = time.time()
            time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
            time_max = time_min + timedelta(
                seconds=self.hparams.time_window_delta_seconds
            )

            # Log the time window we're using
            tplr.logger.info(f"Using time window for gather: {time_min} to {time_max}")

            if self.config.test:
                # In test mode, use all UIDs from metagraph except self
                tplr.logger.info("Test mode active: Using all peers from metagraph.")
                all_uids = list(range(1, len(self.comms.metagraph.S)))
                self.comms.peers = [uid for uid in all_uids if uid != self.uid]

            tplr.logger.info(f"Final peers for gather: {self.comms.peers}")

            gather_result = None
            gather_time = 0.0
            should_update = True

            if self.is_master:
                gather_start = tplr.T()
                tplr.logger.info("Waiting on gather task...")
                gather_result = await self.comms.gather_with_reserve(
                    my_uid=self.uid,
                    gather_uids=self.comms.peers,
                    reserve_uids=self.comms.reserve_peers,
                    window=step_window,
                    key="gradient",
                    timeout=90,
                    device=str(self.device),
                    local=False,
                    stale_retention=100,
                    totalks=self.totalks,
                    compressor=self.compressor,
                    time_min=time_min,
                    time_max=time_max,
                    expected_compressed_params=self.expected_compressed_params,
                )
                tplr.logger.info("Gather task completed!")
                gather_time = tplr.T() - gather_start
                should_update = gather_result is not None

            # Broadcast whether we should update to all ranks
            should_update = dist_helper.all_ok(
                should_update, self.device, "gather_update"
            )

            # 5. Calculate and log metrics
            self.total_tokens_processed += window_tokens
            tokens_per_sec = window_tokens / training_time if training_time else 0.0

            # ---------------------------------------------------------------------
            # 6. Await both gather
            # ---------------------------------------------------------------------

            # 8. Apply gathered gradients
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            update_start = tplr.T()

            # Only perform outer step and increment counter if we have gradients to apply
            stats = None
            if should_update:
                stats = self.outer_step(gather_result)
                self.global_step += (
                    1  # Increment only when we actually do an outer step
                )
                model_update_time = tplr.T() - update_start
                if stats is not None and self.is_master:
                    tplr.logger.info(
                        f"{tplr.P(step_window, model_update_time)} Updated model (Outer step #{self.global_step}) | "
                        f"GradNorm: {stats.get('global_l2_norm', 0.0):.4f}, "
                        f"Params: {len(stats.get('param_norms', {}))}"
                    )
                else:
                    tplr.logger.info(
                        f"{tplr.P(step_window, model_update_time)} Updated model (Outer step #{self.global_step})"
                    )
            else:
                model_update_time = 0.0
                tplr.logger.info(
                    f"{tplr.P(step_window, 0)} Skipped outer step (no gradients gathered)"
                )

            # [TP] Debug sampling avoids collectives: master samples local shards via to_local(), non-masters skip
            debug_dict = {}

            # Add model parameters debug info (sample from local shard to avoid collectives)
            for name, param in self.model.named_parameters():
                if param is not None and self.is_master:
                    # Use to_local() for DTensor to avoid full_tensor() collective deadlock
                    if isinstance(param, DT):
                        flat = param.to_local().data.flatten()
                    else:
                        flat = param.data.flatten()

                    if flat.numel() > 0:
                        # Sample last 2 elements to be consistent with comparison
                        sample = flat[-2:] if flat.numel() >= 2 else flat[-1:]
                        debug_dict[name + "_debug"] = sample.detach().cpu().tolist()

            # Only master uploads the debug dictionary
            if self.is_master:
                # Add successful peers information
                if gather_result is not None:
                    debug_dict["successful_peers"] = sorted(
                        list(set(self.comms.peers) - set(gather_result.skipped_uids))
                    )
                    debug_dict["skipped_peers"] = sorted(
                        list(gather_result.skipped_uids)
                    )

                # Store the debug dictionary
                await self.comms.put(
                    state_dict=debug_dict,
                    uid=str(self.uid),
                    window=step_window,
                    key="debug",
                    local=False,
                )

                tplr.logger.info(
                    f"Stored debug values for window {self.current_window}"
                )

            # Log total window time and metrics
            tplr.logger.info(
                f"{tplr.P(self.current_window, tplr.T() - window_start)} Completed window iteration"
            )

            # ─────────────── momentum norms (gathered across ranks) ─────────
            local_mom_norms: list[float] = [
                m.norm().item() for m in self.error_feedback.values()
            ]
            gathered_mom = dist_helper.all_gather_object(local_mom_norms)

            # Track error feedback memory usage (every 10 windows)
            if self.current_window % 10 == 0:
                gpu_memory = 0
                cpu_memory = 0
                for name in self.error_feedback:
                    if self.error_feedback[name] is not None:
                        if self.error_feedback[name].is_cuda:
                            gpu_memory += (
                                self.error_feedback[name].numel()
                                * self.error_feedback[name].element_size()
                            )
                        else:
                            cpu_memory += (
                                self.error_feedback[name].numel()
                                * self.error_feedback[name].element_size()
                            )

                cpu_buffer_memory = sum(
                    buf.numel() * buf.element_size()
                    for buf in self.error_feedback_cpu_buffers.values()
                )

                tplr.logger.info(
                    f"[Memory] Error feedback: GPU={gpu_memory / (1024**3):.2f}GB, "
                    f"CPU={cpu_memory / (1024**3):.2f}GB, "
                    f"CPU buffers={cpu_buffer_memory / (1024**3):.2f}GB"
                )

            momentum_norms = []
            # Log metrics to WandB
            if self.is_master:
                # Calculate common metrics values
                momentum_norms: list[float] = sum(gathered_mom, [])
                mean_momentum_norm = (
                    sum(momentum_norms) / len(momentum_norms) if momentum_norms else 0
                )
                window_total_time = tplr.T() - window_start
                gather_success_rate = (
                    gather_result.success_rate * 100 if gather_result else 0.0
                )
                inner_lr = self.inner_scheduler.get_last_lr()[0]

                # Only log to WandB when we've performed an outer step
                # This ensures step values are unique and represent actual optimization steps
                if gather_result is not None:
                    wandb_metrics = {
                        # Add timing metrics
                        "miner/timing/window_total": window_total_time,
                        "miner/timing/peer_update": peer_update_time,
                        "miner/timing/data_loading": data_loading_time,
                        "miner/timing/training": training_time,
                        "miner/timing/compression": compression_time,
                        "miner/timing/gather": gather_time,
                        "miner/timing/put": put_time,
                        "miner/timing/model_update": model_update_time,
                        # Existing metrics
                        "miner/window_entry_loss": window_entry_loss,
                        "miner/tokens_per_sec": tokens_per_sec,
                        "miner/total_tokens": self.total_tokens_processed,
                        "miner/batch_tokens": window_tokens,
                        "miner/global_step": self.global_step,
                        "miner/gpu_memory_allocated": torch.cuda.memory_allocated()
                        / 1024**2,
                        "miner/gpu_memory_cached": torch.cuda.memory_reserved()
                        / 1024**2,
                        "miner/gather_peers": len(self.comms.peers),
                        "miner/effective_batch_size": len(self.comms.peers)
                        * self.hparams.batch_size,
                        "miner/inner_lr": inner_lr,
                        # Global gradient and weight norms (now computed correctly)
                        "miner/global_grad_norm": global_grad_norm,
                        "miner/global_weight_norm": global_weight_norm,
                        "miner/mean_momentum_norm": mean_momentum_norm,
                        # Added gather success rate in %
                        "miner/gather/success_rate": gather_success_rate,
                    }

                    # Add Adam optimizer metrics (prefixed with miner/)
                    for key, value in adam_metrics.items():
                        wandb_metrics[f"miner/{key}"] = value

                    # Add outer_step gradient statistics if available
                    if stats is not None:
                        wandb_metrics["miner/outer_step/global_grad_norm"] = stats.get(
                            "global_l2_norm", 0.0
                        )
                        wandb_metrics["miner/outer_step/total_elements"] = stats.get(
                            "total_elements", 0
                        )

                    self.wandb.log(wandb_metrics, step=self.global_step)

                self.metrics_logger.log(
                    measurement="training_step_v2",
                    tags={
                        "window": self.current_window,
                        "global_step": self.global_step,
                    },
                    fields={
                        "loss": window_entry_loss,
                        "n_gather_peers": int(len(self.comms.peers)),
                        "gather_success_rate": gather_success_rate,
                        "gather_peers": json.dumps(self.comms.peers),
                        "skipped_peers": json.dumps(
                            gather_result.skipped_uids if gather_result else []
                        ),
                        "window_total_time": window_total_time,
                        "peer_update_time": peer_update_time,
                        "compression_time": compression_time,
                        "gather_time": gather_time,
                        "put_time": put_time,
                        "model_update_time": model_update_time,
                        "tokens_per_sec": tokens_per_sec,
                    },
                )
                tplr.logger.info("Finished metrics logging call for miner")

            # global_step is now incremented only when outer_step is performed
            tplr.logger.info(f"Total outer steps taken: {self.global_step}")

            dist_helper.safe_barrier("post_outer_step", self.local_rank)

            # Delete any remaining local variables to clear up memory
            del shard_gradient
            if gather_result is not None:
                del gather_result
            torch.cuda.empty_cache()
            # Check memory threshold periodically
            self.check_memory_threshold(threshold_gb=0.5)

            await self.cleanup_window()

            # 4. Wait for next window
            tplr.logger.info("Wait for next window...")
            await self.wait_until_window(step_window + 1)

    async def cleanup_window(self):
        """Aggressive memory cleanup between windows"""
        # Clear gradients more thoroughly
        self.model.zero_grad(set_to_none=True)
        self.inner_optimizer.zero_grad(set_to_none=True)

        # Clear error feedback for non-owned params to save memory
        for name in list(self.error_feedback.keys()):
            if name not in self.owned_params and self.error_feedback[name] is not None:
                self.error_feedback[name] = None

        # Clear any cached autocast states
        torch.clear_autocast_cache()

        # Empty CUDA cache multiple times for thorough cleanup
        for _ in range(3):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        # Force garbage collection
        gc.collect()

        # Check memory threshold after cleanup
        self.check_memory_threshold(threshold_gb=1.0)

        # Log memory status
        tplr.logger.info(
            f"After cleanup - GPU allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB"
        )
        tplr.logger.info(
            f"After cleanup - GPU reserved: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB"
        )

    def _reconstruct_tp_gradients(self, tp_shards):
        """
        Reconstruct full gradients from tensor-parallel shards.

        Process Overview:
        1. Collect shards from all TP ranks
        2. Decompress each shard (sparse → 4D tensor)
        3. Decode to 2D without DCT (lossless unchunking)
        4. Concatenate shards along split dimension
        5. Encode back to 4D without DCT (lossless chunking)
        6. Recompress to sparse format for validator

        Args:
            tp_shards: List of gradient dictionaries from TP ranks

        Returns:
            Dictionary with reconstructed full gradients maintaining compression
        """
        if not tp_shards or not any(tp_shards):
            return {}

        # Get all parameter names from first non-None shard
        all_params = set()
        for shard in tp_shards:
            if shard:
                for key in shard:
                    if key.endswith("idxs"):
                        all_params.add(key[:-4])  # Remove "idxs" suffix
                break

        result = {}

        # Process each parameter
        for param_name in all_params:
            # Check if this is a TP-sharded parameter
            first_shard = next(s for s in tp_shards if s)
            shard_meta = first_shard.get(param_name + "shard_metadata", {})

            if shard_meta.get("is_tp_sharded"):
                # Collect all TP shards for this parameter
                shards_data = []
                for rank_idx, shard_dict in enumerate(tp_shards):
                    if shard_dict and (param_name + "idxs") in shard_dict:
                        shards_data.append(
                            {
                                "rank": rank_idx,
                                "idxs": shard_dict.get(param_name + "idxs"),
                                "vals": shard_dict.get(param_name + "vals"),
                                "quant_params": shard_dict.get(
                                    param_name + "quant_params"
                                ),
                                "xshape": shard_dict.get("metadata", {})
                                .get("xshapes", {})
                                .get(param_name),
                                "totalk": shard_dict.get("metadata", {})
                                .get("totalks", {})
                                .get(param_name),
                                "local_shape": shard_meta.get("local_shape"),
                                "global_shape": shard_meta.get("global_shape"),
                                "shard_dim": shard_meta.get("shard_dim", 0),
                            }
                        )

                if not shards_data:
                    continue

                # TP Gradient Reconstruction Process
                # Why we decompress and recompress:
                # - Each TP rank sends a shard of the gradient
                # - We need to combine shards into a full gradient for the validator
                # - The validator expects a single compressed gradient, not multiple shards
                # Important: We must use the same use_dct setting as the original compression
                # to ensure correct reconstruction without corruption

                shard_dim = shards_data[0]["shard_dim"]
                global_shape = shards_data[0]["global_shape"]

                # Get use_dct setting to match original encoding
                use_dct = getattr(self.hparams, "use_dct", False)

                # Step 1: DECOMPRESS each shard (compressed format → encoded tensor)
                decompressed_shards = []
                for shard in shards_data:
                    if shard["idxs"] is None or shard["vals"] is None:
                        continue

                    # Use xshape from compression metadata
                    shard_xshape = shard["xshape"] or shard["local_shape"]
                    shard_totalk = shard["totalk"] or len(shard["idxs"])

                    # Move quantization params to device if present
                    quant_params_on_device = None
                    if shard["quant_params"] is not None:
                        if isinstance(shard["quant_params"], dict):
                            quant_params_on_device = {}
                            for key, value in shard["quant_params"].items():
                                if isinstance(value, torch.Tensor):
                                    quant_params_on_device[key] = value.to(self.device)
                                else:
                                    quant_params_on_device[key] = value
                        elif isinstance(shard["quant_params"], (tuple, list)):
                            quant_params_on_device = []
                            for item in shard["quant_params"]:
                                if isinstance(item, torch.Tensor):
                                    quant_params_on_device.append(item.to(self.device))
                                else:
                                    quant_params_on_device.append(item)
                            quant_params_on_device = tuple(quant_params_on_device)
                        else:
                            quant_params_on_device = shard["quant_params"]

                    # Create reference tensor for decompression
                    ref = torch.empty(
                        shard_xshape, device=self.device, dtype=torch.float32
                    )

                    # Decompress the shard (sparse format → encoded 4D tensor)
                    decompressed = self.compressor.decompress(
                        ref,
                        shard["idxs"].to(self.device),
                        shard["vals"].to(self.device),
                        shard_xshape,
                        shard_totalk,
                        quantize_params=quant_params_on_device,
                    )

                    # Decode from transformer space (4D chunked → 2D gradient tensor)
                    # Must use the same DCT setting as original encoding
                    decoded = self.transformer.decode(
                        decompressed,
                        use_dct=use_dct,  # Match original encoding
                    )

                    # Reshape if needed to match local shape
                    if shard["local_shape"] and tuple(decoded.shape) != tuple(
                        shard["local_shape"]
                    ):
                        decoded = decoded.reshape(shard["local_shape"])

                    decompressed_shards.append(decoded)
                    del decompressed  # Free memory

                if not decompressed_shards:
                    continue

                # Step 2: CONCATENATE shards to form the full gradient
                # This combines the TP shards along their split dimension (e.g., columns for MLP)
                full_gradient = torch.cat(decompressed_shards, dim=shard_dim)
                num_shards = len(decompressed_shards)  # Store for metadata

                # Step 3: RECOMPRESS the full gradient for the validator
                # Must use the same DCT setting to maintain consistency
                encoded = self.transformer.encode(
                    full_gradient,
                    use_dct=use_dct,  # Match original encoding
                )

                # Compress the full gradient (4D tensor → sparse format)
                topk_compression = getattr(self.hparams, "topk_compression", 32)
                compress_result = self.compressor.compress(encoded, topk_compression)

                # Handle both 3-tuple and 5-tuple returns from compress
                if len(compress_result) == 3:
                    idxs, vals, xshape = compress_result
                    totalk = len(idxs)
                    quant_params = None
                elif len(compress_result) == 5:
                    idxs, vals, xshape, totalk, quant_params = compress_result
                else:
                    raise ValueError(
                        f"Unexpected compress return: {len(compress_result)} values"
                    )

                # Store the recompressed full gradient
                result[param_name + "idxs"] = idxs
                result[param_name + "vals"] = vals
                if quant_params:
                    result[param_name + "quant_params"] = quant_params
                # Store metadata about this reconstructed parameter
                # This tells the validator that this gradient:
                # - Is a full gradient (not a shard)
                # - Was reconstructed from TP shards
                # - Has specific shape/compression parameters
                result[param_name + "shard_metadata"] = {
                    "is_shard": False,  # This is now a full gradient
                    "was_tp_sharded": True,  # Was originally TP sharded
                    "num_tp_shards_merged": num_shards,  # Number of shards combined
                    "global_shape": tuple(xshape),  # Shape for decompression
                    "totalk": totalk,  # Number of values kept after compression
                }

                # Free GPU memory immediately
                del decompressed_shards, full_gradient, encoded

            else:
                # Not TP-sharded (or replicated), just copy from first shard that has it
                for shard_dict in tp_shards:
                    if shard_dict and (param_name + "idxs") in shard_dict:
                        result[param_name + "idxs"] = shard_dict[param_name + "idxs"]
                        result[param_name + "vals"] = shard_dict[param_name + "vals"]
                        if param_name + "quant_params" in shard_dict:
                            result[param_name + "quant_params"] = shard_dict[
                                param_name + "quant_params"
                            ]
                        result[param_name + "shard_metadata"] = shard_dict.get(
                            param_name + "shard_metadata", {}
                        )
                        break

        # Copy and update metadata from first shard
        first_shard = next((s for s in tp_shards if s), {})
        if "metadata" in first_shard:
            result["metadata"] = first_shard["metadata"].copy()
            # Update xshapes and totalks for reconstructed parameters
            for param_name in all_params:
                if (param_name + "shard_metadata") in result:
                    meta = result[param_name + "shard_metadata"]
                    if meta.get("was_tp_sharded") and meta.get("global_shape"):
                        # Update shapes for reconstructed parameters
                        if "xshapes" not in result["metadata"]:
                            result["metadata"]["xshapes"] = {}
                        if "totalks" not in result["metadata"]:
                            result["metadata"]["totalks"] = {}
                        # Use the global shape for reconstructed TP parameters
                        result["metadata"]["xshapes"][param_name] = meta["global_shape"]
                        # Use the actual totalk from reconstruction metadata
                        result["metadata"]["totalks"][param_name] = meta.get(
                            "totalk",
                            len(result[param_name + "idxs"])
                            if param_name + "idxs" in result
                            else 0,
                        )

        return result


# Start miner.
if __name__ == "__main__":
    uvloop.install()
    try:
        asyncio.run(Miner().main())
    except KeyboardInterrupt:
        pass
