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


"""Efficient evaluator leveraging existing Templar infrastructure.

Key features:
- Uses existing comms and DCPCheckpointer
- Proper multi-GPU evaluation with vllm
- One-time baseline evaluation
- Efficient checkpoint discovery without downloading
- Model caching to avoid redundant conversions
"""

import argparse
import asyncio
import contextlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import bittensor as bt
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchtitan.components.loss import cross_entropy_loss
from tqdm.auto import tqdm

import tplr
from tplr.distributed import dist_helper
from tplr.model_factory import (
    convert_titan_to_hf,
    create_meta_model,
    initialize_torchtitan_model,
)


def _cuda_gc(label: str = "") -> None:
    """Aggressively release CUDA memory (and pinned host pools)."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Collect inter-process handles, then clear allocator caches
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception:
        pass
    if label:
        tplr.logger.info(f"[mem] CUDA cleanup done: {label}")


@contextlib.contextmanager
def pause_ddp_for_lm_eval(tag: str):
    """
    Context manager that synchronizes ranks for lm-eval execution.

    1) All ranks barrier on entry
    2) Rank 0 runs lm-eval while others wait on sentinel file
    3) All ranks barrier on exit after cleanup
    """
    if dist_helper.is_distributed():
        dist_helper.safe_barrier(
            tag=f"pause_ddp_enter_{tag}", local_rank=dist_helper.local_rank
        )

    sentinel = Path(f"/tmp/tplr_eval_done_{tag}")
    try:
        yield sentinel
    finally:
        if not dist_helper.is_master:
            while not sentinel.exists():
                time.sleep(0.25)

        if dist_helper.is_distributed():
            dist_helper.safe_barrier(
                tag=f"pause_ddp_exit_{tag}", local_rank=dist_helper.local_rank
            )

        if dist_helper.is_master and sentinel.exists():
            try:
                sentinel.unlink()
                tplr.logger.info("[Master] Cleaned up sentinel file")
            except Exception:
                pass


LN2 = math.log(2.0)


class NullMetricsLogger:
    """Null logger for non-master ranks."""

    def log(self, *_args, **_kwargs) -> None:
        return


def _loss_to_bpb(
    total_loss_nats: float,
    total_tokens: int,
    *,
    total_bytes: int | None = None,
    bytes_per_token: float | None = None,
) -> float:
    """
    Convert accumulated cross-entropy (nats) to bits-per-byte (bpb).
    Prefer `total_bytes` for exactness; `bytes_per_token` is optional fallback.
    """
    if total_tokens <= 0:
        return float("inf")
    avg_loss_nats = total_loss_nats / total_tokens
    bits_per_token = avg_loss_nats / LN2
    if total_bytes is not None and total_bytes > 0:
        tokens_per_byte = total_tokens / total_bytes
        return bits_per_token * tokens_per_byte
    if bytes_per_token is not None and bytes_per_token > 0:
        tokens_per_byte = 1.0 / bytes_per_token
        return bits_per_token * tokens_per_byte
    raise ValueError("Provide total_bytes or bytes_per_token to compute bpb.")


class ModelCache:
    """Manages converted HuggingFace models."""

    def __init__(
        self, base_dir: Path = Path("models/cache"), namespace: str | None = None
    ):
        # Scope the cache by version (namespace) to avoid collisions when
        # the same window number exists across different model versions.
        if namespace:
            self.base_dir = base_dir / namespace
        else:
            self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, window: int) -> Path:
        """Get path for a cached model."""
        return self.base_dir / f"window_{window}"

    def exists(self, window: int) -> bool:
        """Check if model is cached."""
        path = self.get_path(window)
        if not path.exists():
            return False

        # Check for HF model files
        has_config = (path / "config.json").exists()
        has_weights = any(
            [
                (path / "model.safetensors").exists(),
                len(list(path.glob("model-*.safetensors"))) > 0,
                (path / "pytorch_model.bin").exists(),
            ]
        )

        return has_config and has_weights

    def cleanup(self, keep_latest: int = 2):
        """Remove old cached models."""
        # Get all window directories and extract window numbers
        model_dirs = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("window_"):
                try:
                    window_num = int(d.name.split("_")[1])
                    model_dirs.append((window_num, d))
                except (IndexError, ValueError):
                    # Skip directories that don't match expected format
                    continue

        # Sort by window number (ascending)
        model_dirs.sort(key=lambda x: x[0])

        # Remove all but the latest keep_latest models
        if len(model_dirs) > keep_latest:
            for _, old_model in model_dirs[:-keep_latest]:
                try:
                    shutil.rmtree(old_model)
                    tplr.logger.info(f"Removed old model cache: {old_model}")
                except Exception as e:
                    tplr.logger.warning(f"Failed to remove {old_model}: {e}")
        else:
            tplr.logger.info(
                f"Model cache has {len(model_dirs)} models, keeping all (threshold: {keep_latest})"
            )


class Evaluator:
    """Evaluator using existing Templar infrastructure."""

    # Track active subprocess for cleanup on termination
    _active_subprocess: subprocess.Popen | None = None

    def __init__(self, config: bt.config):
        self.config = config

        # Core configuration
        self.netuid = config.netuid
        self.version = config.version or tplr.__version__
        tplr.__version__ = self.version
        tplr.logger.info(
            f"Running evaluations for {self.version = }. Setting tplr version to {tplr.__version__}"
        )
        self.eval_interval = config.eval_interval

        # Backfill & control flags
        self.from_window: int | None = config.from_window
        self.to_window: int | None = config.to_window
        self.no_follow: bool = bool(config.no_follow)
        self.force_reval: bool = bool(config.force_reval)

        # Load hyperparameters
        self.hparams = tplr.load_hparams()
        tplr.logger.info(f"Loaded hparams for {self.hparams.model_size}")

        # Initialize distributed using the helper
        dist_helper.init_process_group()
        self.rank = dist_helper.rank
        self.world_size = dist_helper.world_size
        self.local_rank = dist_helper.local_rank
        self.is_master = dist_helper.is_master
        assert dist_helper.device is not None
        self.device = dist_helper.device

        if self.world_size < 2:
            raise ValueError("Evaluator requires multi-GPU setup (WORLD_SIZE >= 2)")

        # Setup comms with full functionality
        self.comms = tplr.comms.Comms(
            wallet=None,
            save_location="/tmp",
            key_prefix="checkpoint",
            config=self.config,
            hparams=self.hparams,
            uid=None,  # Not needed for evaluator
        )

        # Setup checkpoint manager first
        self.uid = 1  # Fixed UID for evaluator
        self.ckpt = tplr.DCPCheckpointer(
            self.comms, uid=self.uid, version=self.version, repo_root="."
        )

        # Initialize WandB and metrics logger on master rank
        if self.is_master:
            self.wandb = tplr.initialize_wandb(
                run_prefix="E",
                uid=str(self.uid),
                config=self.config,
                group="evaluator",
                job_type="evaluation",
            )
            self.metrics_logger = tplr.metrics.MetricsLogger(
                prefix="E",
                uid=str(self.uid),
                config=self.config,
                role="evaluator",
                group="evaluations",
                job_type="eval",
                version=self.version,
            )
        else:
            self.wandb = NullMetricsLogger()
            self.metrics_logger = NullMetricsLogger()

        # State tracking
        self.evaluated_windows: set[int] = set()
        self.baseline_evaluated = False
        self.start_window: int = 0  # Track start window for global_step calculation

        # Initialize or load model
        self.model = self._initialize_or_load_model()

        # Model cache
        self.model_cache = ModelCache(
            Path(cast(str, config.cache_dir)), namespace=self.version
        )

        # Tasks configuration
        self.tasks = (
            config.tasks.split(",")
            if config.tasks
            else ["arc_challenge", "arc_easy", "hellaswag", "winogrande"]
        )

        # Tokenizer
        self.tokenizer = self.hparams.tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _initialize_or_load_model(self):
        """Initialize a new model or load from local checkpoint if available."""
        # Check for existing local checkpoint
        checkpoint_dir = Path("checkpoints") / self.version
        local_checkpoint_window = None

        if checkpoint_dir.exists():
            # Find the latest window directory
            window_dirs = [
                d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if window_dirs:
                latest_window_dir = max(window_dirs, key=lambda d: int(d.name))
                local_checkpoint_window = int(latest_window_dir.name)
                if self.is_master:
                    tplr.logger.info(
                        f"Found local checkpoint at window {local_checkpoint_window}"
                    )

        if local_checkpoint_window is not None:
            # Create model on meta device for fast loading
            model = create_meta_model(
                hparams=self.hparams,
                role="evaluator",
                world_size=self.world_size,
            )

            # Move model from meta to actual device (allocates memory but no initialization)
            model.to_empty(device=str(self.device))

            try:
                if self.is_master:
                    tplr.logger.info(
                        f"Loading local checkpoint from window {local_checkpoint_window}"
                    )
                self.ckpt.load_local(
                    model=model, window=local_checkpoint_window, process_group=None
                )
                self.evaluated_windows.add(
                    local_checkpoint_window
                )  # Mark as already evaluated
                if self.is_master:
                    tplr.logger.info(
                        f"Successfully loaded checkpoint from window {local_checkpoint_window}"
                    )
                self.baseline_evaluated = True
                return model
            except Exception as e:
                tplr.logger.warning(
                    f"Failed to load local checkpoint: {e}. Initializing new model."
                )
                # Fall back to initialized model
                return initialize_torchtitan_model(
                    hparams=self.hparams,
                    role="evaluator",
                    device=str(self.device),
                    world_size=self.world_size,
                )
        else:
            # No checkpoint found, initialize new model
            if self.is_master:
                tplr.logger.info("No local checkpoint found. Initializing new model.")
            return initialize_torchtitan_model(
                hparams=self.hparams,
                role="evaluator",
                device=str(self.device),
                world_size=self.world_size,
            )

    async def check_latest_checkpoint(self) -> int | None:
        """Check for latest checkpoint without downloading."""
        ready_window: int | None = None

        if self.is_master:
            try:
                candidate = await self.ckpt._discover_latest(prefer_highest_staked=True)

                # Gate on completeness
                if candidate is not None and candidate not in self.evaluated_windows:
                    is_complete = await self.ckpt.check_checkpoint_exists(
                        window=candidate
                    )
                    if is_complete:
                        tplr.logger.info(f"New checkpoint READY at window {candidate}")
                        ready_window = candidate
                    else:
                        tplr.logger.info(
                            f"Checkpoint pointer found for window {candidate}, "
                            "but upload not complete yet; will retry."
                        )
            except Exception:
                tplr.logger.exception("Checkpoint discovery")

        # Only broadcast a window if it is READY
        tensor_val = -1 if ready_window is None else ready_window
        window_tensor = torch.tensor(
            [tensor_val],
            dtype=torch.int32,
            device=self.device if self.device != "cpu" else "cpu",
        )
        dist_helper.broadcast(window_tensor, src=0)

        value = int(window_tensor.item())
        return None if value < 0 else value

    def save_model_for_eval(self, window: int) -> Path | None:
        """Convert and save model in HuggingFace format."""

        # Check cache first
        if self.model_cache.exists(window):
            tplr.logger.info(f"Using cached model for window {window}")
            return self.model_cache.get_path(window)

        model_path = self.model_cache.get_path(window)
        model_path.mkdir(parents=True, exist_ok=True)

        try:
            tplr.logger.info(f"Converting model for window {window}...")

            # Convert using existing function
            convert_titan_to_hf(
                titan_model=self.model,
                hparams=self.hparams,
                save_path=str(model_path),
                is_master=self.is_master,
            )

            # Save tokenizer on master
            if self.is_master:
                self.tokenizer.save_pretrained(str(model_path))

            # Sync all ranks
            dist_helper.safe_barrier(tag="model_save", local_rank=self.local_rank)

            tplr.logger.info(f"Model saved to {model_path}")
            return model_path

        except Exception as e:
            tplr.logger.error(f"Model conversion failed: {e}")
            if model_path.exists():
                shutil.rmtree(model_path)
            return None

    def run_lm_eval_multi_gpu(
        self,
        model_path: Path,
        tasks: list[str],
        window: int,
        global_step: int,
        num_fewshot: int = 0,
    ) -> dict[str, float]:
        output_dir = Path(f"/tmp/eval_{int(time.time())}")
        output_dir.mkdir(parents=True, exist_ok=True)

        tp = self.world_size  # or torch.cuda.device_count() if you choose
        cmd = [
            "uvx",
            "--with",
            "lm_eval[vllm]",
            "lm_eval",
            "--model",
            "vllm",
            "--model_args",
            f"pretrained={model_path},tensor_parallel_size={tp},gpu_memory_utilization=0.85",
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            "auto",
            "--output_path",
            str(output_dir),
        ]
        if num_fewshot > 0:
            cmd.extend(["--num_fewshot", str(num_fewshot)])

        # ── NEW: clean environment for the child ──────────────────────────────
        clean_env = os.environ.copy()
        for k in [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_NAME",
            "OMP_NUM_THREADS",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_USE_AGENT_STORE",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_RUN_ID",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "TORCHELASTIC_ERROR_FILE",
            "NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_SOCKET_IFNAME",
            # Also remove NCCL-specific environment variables
            "NCCL_DEBUG",
            "NCCL_DEBUG_SUBSYS",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
            "NCCL_TREE_THRESHOLD",
        ]:
            clean_env.pop(k, None)

        # Make sure child sees all GPUs (master rank only spawns this)
        if torch.cuda.is_available():
            clean_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(torch.cuda.device_count())
            )
        # Safer worker startup for vLLM
        clean_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        tplr.logger.info(f"[Master] Running lm-eval: {' '.join(cmd)}")
        start = time.perf_counter()

        self._active_subprocess = subprocess.Popen(
            cmd,
            stdout=None,  # inherit parent stdout (no per-line piping)
            stderr=None,  # inherit parent stderr
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=clean_env,  # ← IMPORTANT
        )
        process = self._active_subprocess

        tplr.logger.info("[Master] Waiting for lm-eval subprocess to complete...")
        return_code = process.wait()
        self._active_subprocess = None  # Clear reference after completion

        runtime = time.perf_counter() - start

        # Note: vllm may exit with 134 (SIGABRT) after successful completion
        if return_code not in [0, 134]:
            tplr.logger.error(f"[Master] lm-eval failed with return code {return_code}")
            # Don't cleanup on failure so we can inspect
            return {}
        elif return_code == 134:
            tplr.logger.warning(
                f"[Master] lm-eval exited with code 134 (SIGABRT), checking for results anyway..."
            )

        tplr.logger.info(
            f"[Master] lm-eval completed in {runtime:.2f}s, parsing results..."
        )

        # Parse results using the dedicated method
        results = self.parse_lm_eval_results(
            output_dir, window=window, global_step=global_step
        )

        # Cleanup only on success
        if results:
            shutil.rmtree(output_dir, ignore_errors=True)
            tplr.logger.info(f"[Master] Cleaned up {output_dir}")
        else:
            tplr.logger.warning(
                f"[Master] No results parsed, keeping {output_dir} for inspection"
            )

        tplr.logger.info(
            f"[Master] Evaluation parsing completed, found {len(results)} task results"
        )

        return results

    def parse_lm_eval_results(
        self, output_dir: Path, window: int, global_step: int
    ) -> dict[str, float]:
        """Parse lm-eval results from output directory and log metrics.

        Args:
            output_dir: Directory containing lm-eval output files
            window: Current window number for metrics logging
            global_step: Global step for metrics logging

        Returns:
            Dictionary mapping task names to accuracy scores
        """
        results = {}

        # Debug: List all files in output dir
        tplr.logger.debug(f"[Master] Searching for results in {output_dir}")
        all_files = list(output_dir.rglob("*"))
        tplr.logger.debug(f"[Master] Found {len(all_files)} total files/dirs")
        for f in all_files[:10]:  # Show first 10
            tplr.logger.debug(
                f"  - {f.relative_to(output_dir)} (is_file={f.is_file()})"
            )

        # Look for JSON files in subdirectories (lm-eval creates a subdir based on model name)
        json_files = list(output_dir.glob("**/*.json"))

        if not json_files:
            # Fallback to root directory
            json_files = list(output_dir.glob("*.json"))

        if json_files:
            tplr.logger.debug(f"[Master] Found {len(json_files)} JSON file(s):")
            for jf in json_files:
                tplr.logger.debug(f"  - {jf.relative_to(output_dir)}")

            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            tplr.logger.info(f"[Master] Reading results from {latest_file}")

            try:
                with open(latest_file) as f:
                    data = json.load(f)

                tplr.logger.debug(
                    f"[Master] JSON structure: top-level keys = {list(data.keys())}"
                )

                if "results" in data:
                    for task_name, task_results in data.get("results", {}).items():
                        tplr.logger.debug(
                            f"  Task '{task_name}' metrics: {list(task_results.keys())}"
                        )

                        # Use the same metric priority as old evaluator
                        # see: https://github.com/EleutherAI/lm-evaluation-harness/blob/758c5ed891b1ca48acd8d3a0d309a827215796b7/scripts/regression.py#L115
                        metric_names = ["acc_norm,none", "acc,none"]
                        metric_value = None
                        used_metric = None

                        for metric_name in metric_names:
                            if (value := task_results.get(metric_name)) is not None:
                                metric_value = value
                                used_metric = metric_name
                                break

                        if metric_value is not None:
                            results[task_name] = metric_value
                            tplr.logger.info(
                                f"  ✓ {task_name} ({used_metric}): {metric_value:.4f}"
                            )

                            # Log to metrics like the old evaluator
                            self.metrics_logger.log(
                                measurement="benchmark_task",
                                tags={
                                    "task": task_name,
                                    "metric": used_metric,
                                    "global_step": global_step,
                                    "window": window,
                                },
                                fields={"score": float(metric_value)},
                            )
                            # Log to wandb
                            self.wandb.log(
                                {
                                    f"evaluator/benchmark/{task_name}/{used_metric}": metric_value,
                                },
                                step=global_step,
                            )
                        else:
                            tplr.logger.warning(
                                f"  ✗ {task_name}: no acc_norm,none or acc,none found"
                            )

                    # Log summary metrics like old evaluator
                    self.metrics_logger.log(
                        measurement="benchmark_summary",
                        tags={
                            "global_step": global_step,
                            "window": window,
                        },
                        fields={
                            "num_tasks": len(results),
                            "global_step": global_step,
                        },
                    )
                    # Log summary to wandb - commit to push metrics immediately
                    self.wandb.log(
                        {
                            "evaluator/benchmark/num_tasks": len(results),
                        },
                        step=global_step,
                        commit=True,
                    )

            except Exception as e:
                tplr.logger.error(f"[Master] Failed to parse {latest_file}: {e}")
                tplr.logger.error(f"[Master] Keeping {output_dir} for inspection")
                return {}
        else:
            tplr.logger.warning(f"[Master] No result files found in {output_dir}")

        return results

    def run_custom_eval(
        self, window: int, global_step: int
    ) -> tuple[float, float] | None:
        """Run custom dataset evaluation, returns (perplexity, loss)."""
        if not self.config.custom_eval_path:
            return None

        tplr.logger.info("Running custom perplexity evaluation")

        # Track evaluation time
        start_time = time.perf_counter()

        # Setup dataset
        os.environ["DATASET_BINS_PATH"] = self.config.custom_eval_path
        eval_path = Path(self.config.custom_eval_path)

        # Find file prefix
        file_prefix = "train"
        for prefix in ["val", "eval", "test"]:
            if (eval_path / f"{prefix}_000000.npy").exists():
                file_prefix = prefix
                break

        # Create dataset with distributed sampling
        dataset = tplr.SharedShardedDataset(
            shard_index=0,
            sequence_length=self.hparams.sequence_length,
            rank=self.rank,
            world_size=self.world_size,
            file_prefix=file_prefix,
        )

        # Distributed sampling
        total_samples = min(1024, len(dataset))
        if self.world_size > 1:
            samples_per_rank = total_samples // self.world_size
            start_idx = self.rank * samples_per_rank
            end_idx = (
                start_idx + samples_per_rank
                if self.rank < self.world_size - 1
                else total_samples
            )
            indices = list(range(start_idx, end_idx))
        else:
            indices = list(range(total_samples))

        sampler = torch.utils.data.SubsetRandomSampler(indices)
        # For eval we avoid lingering worker/pinned memory to keep VRAM clean
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=0,  # no workers → no worker lifetime/pin pools
            pin_memory=False,  # avoid pinned host buffers
        )

        # Run evaluation
        self.model.eval()
        local_loss = 0.0
        local_tokens = 0
        local_bytes = 0

        with torch.inference_mode():
            for batch in tqdm(
                dataloader, desc="Custom Eval", disable=not self.is_master
            ):
                input_ids = batch.to(self.device, dtype=torch.long, non_blocking=False)
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]  # ✓ shift by +1
                labels[:, -1] = self.tokenizer.pad_token_id
                labels = torch.where(
                    labels == self.tokenizer.pad_token_id, -100, labels
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids, labels)

                loss = cross_entropy_loss(logits, labels)
                num_tokens = (labels != -100).sum().item()

                if num_tokens > 0:
                    local_loss += loss.item() * num_tokens
                    local_tokens += int(num_tokens)

                # Exact bytes via tokenizer decode (same tokenizer as model).
                # Decode only the evaluated sequence (input_ids) and count UTF-8 bytes.
                # batch_decode expects python lists, not tensors.
                ids_list = input_ids.detach().cpu().tolist()
                texts = self.tokenizer.batch_decode(
                    ids_list,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                local_bytes += sum(len(t.encode("utf-8")) for t in texts)

                # Drop references ASAP to let the allocator reuse/freed blocks
                del logits, labels, input_ids
        # Encourage allocator to release caches after eval
        _cuda_gc("post-custom-eval")

        # All-reduce across ranks
        if self.world_size > 1:
            total_loss = dist_helper.ddp_reduce(
                local_loss, op=dist.ReduceOp.SUM, device=self.device
            )
            total_tokens = int(
                dist_helper.ddp_reduce(
                    local_tokens, op=dist.ReduceOp.SUM, device=self.device
                )
            )
            total_bytes = int(
                dist_helper.ddp_reduce(
                    local_bytes, op=dist.ReduceOp.SUM, device=self.device
                )
            )
        else:
            total_loss = local_loss
            total_tokens = local_tokens
            total_bytes = local_bytes

        average_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = (
            torch.exp(torch.tensor(average_loss)).item()
            if total_tokens > 0
            else float("inf")
        )

        # Bits per byte (exact): use total_bytes from decoded sequences.
        bpb = _loss_to_bpb(total_loss, total_tokens, total_bytes=total_bytes)

        # Calculate runtime
        eval_runtime = time.perf_counter() - start_time

        if self.is_master:
            tplr.logger.info("Custom evaluation results:")
            tplr.logger.info(f"  ✓ Perplexity: {perplexity:.4f}")
            tplr.logger.info(f"  ✓ Avg Loss: {average_loss:.4f}")
            tplr.logger.info(
                f"  ✓ Bits per byte (exact): {bpb:.4f}  [tokens={total_tokens}, bytes={total_bytes}]"
            )
            tplr.logger.info(f"  ✓ Eval Time: {eval_runtime:.2f}s")

            # Log metrics
            self.metrics_logger.log(
                "custom_evaluation",
                tags={
                    "global_step": global_step,
                    "window": window,
                    "world_size": self.world_size,
                },
                fields={
                    "perplexity": perplexity,
                    "average_loss": average_loss,
                    "bpb": bpb,
                    "runtime_s": eval_runtime,
                    "total_tokens": total_tokens,
                    "total_bytes": total_bytes,
                },
            )
            # Log to wandb
            self.wandb.log(
                {
                    "evaluator/custom_eval/perplexity": perplexity,
                    "evaluator/custom_eval/average_loss": average_loss,
                    "evaluator/custom_eval/bpb": bpb,
                    "evaluator/custom_eval/runtime_s": eval_runtime,
                    "evaluator/custom_eval/total_tokens": total_tokens,
                    "evaluator/custom_eval/total_bytes": total_bytes,
                },
                step=global_step,
            )

        return perplexity, average_loss

    async def evaluate_window(self, window: int, is_baseline: bool = False) -> bool:
        """Evaluate a specific window."""

        if window in self.evaluated_windows and not self.force_reval:
            tplr.logger.info(f"Window {window} already evaluated")
            return True

        tplr.logger.info(
            f"Starting {'baseline' if is_baseline else 'checkpoint'} "
            f"evaluation for window {window}"
        )

        # Load checkpoint if not baseline
        if not is_baseline and window > 0:
            try:
                res = await self.ckpt.download_and_load(
                    model=self.model,
                    window=window,
                    shared_fs=True,
                    process_group=None,
                    prefer_highest_staked=True,
                )

                if res is None:
                    tplr.logger.error(f"Could not download checkpoint for {window}")
                    return False
                loaded_window, global_step = res
                if loaded_window != window:
                    tplr.logger.error(
                        f"Window mismatch: expected {window}, got {loaded_window}"
                    )
                    return False
                if global_step == -1:
                    # Calculate global_step
                    global_step = window - self.start_window if window > 0 else 0
                    tplr.logger.info(
                        "No global step in checkpoint sidecar. "
                        f"Calculating from start window to be {global_step}."
                    )

            except Exception as e:
                tplr.logger.error(f"Failed to load checkpoint: {e}")
                return False
        else:
            # For baseline evaluation, global_step is 0
            global_step = 0

        # Convert and save model
        model_path = self.save_model_for_eval(window)
        if model_path is None:
            return False

        # Ensure non-negative global_step if we are backfilling earlier than start_window
        # (avoid negative steps in metrics backends).
        if isinstance(global_step, int) and global_step < 0:
            global_step = 0

        # Run custom perplexity evaluation
        _ = self.run_custom_eval(window, global_step)

        # Move model to CPU to free GPU memory for lm-eval
        self.model = self.model.to("cpu")

        # Ensure all CUDA memory is cleared (including allocator caches)
        _cuda_gc("pre-lm-eval")

        # Run lm-eval benchmarks inside context manager
        if self.tasks:
            tplr.logger.info(
                f"[Rank {self.rank}/{self.world_size}] Entering lm-eval context for window {window}"
            )
            with pause_ddp_for_lm_eval(f"win{window}") as sentinel:
                if self.is_master:
                    try:
                        tplr.logger.info(
                            f"[Master] Running evaluation for tasks: {self.tasks}"
                        )
                        results = self.run_lm_eval_multi_gpu(
                            model_path=model_path,
                            tasks=self.tasks,
                            window=window,
                            global_step=global_step,
                        )
                        tplr.logger.info(f"[Master] Task results: {results}")
                    finally:
                        # Always signal completion so non-masters can proceed
                        tplr.logger.info(
                            "[Master] Signaling lm-eval completion to other ranks"
                        )
                        try:
                            sentinel.touch()
                        except Exception as e:
                            tplr.logger.error(
                                f"[Master] Failed to create sentinel: {e}"
                            )

        # Move model back to GPU
        self.model = self.model.to(self.device)

        # Mark as evaluated
        self.evaluated_windows.add(window)

        if self.is_master:
            tplr.logger.info(
                f"[Master] Window {window} evaluation complete. Total evaluated: {len(self.evaluated_windows)}"
            )

        # Cleanup old models and checkpoints
        if len(self.evaluated_windows) >= 1:
            if self.is_master:
                tplr.logger.info(
                    f"[Master] Cleaning up old model caches and checkpoints (keeping latest 1)"
                )
                self.model_cache.cleanup(keep_latest=1)
                self.ckpt.cleanup_local_checkpoints(keep_latest=1)
        return True

    async def _resolve_to_window(self) -> int | None:
        """
        Resolve the 'to_window' bound when backfilling if it wasn't provided.
        Uses the remote pointer to the latest complete checkpoint.
        """
        latest: int | None = None
        if self.is_master:
            try:
                candidate = await self.ckpt._discover_latest(prefer_highest_staked=True)
                if candidate is not None:
                    is_complete = await self.ckpt.check_checkpoint_exists(
                        window=candidate
                    )
                    if is_complete:
                        latest = candidate
                        tplr.logger.info(
                            f"[Master] Resolved to_window → latest READY window {latest}"
                        )
            except Exception:
                tplr.logger.exception("Resolving to_window (latest)")
        # broadcast
        t = torch.tensor(
            [-1 if latest is None else latest],
            dtype=torch.int32,
            device=self.device if self.device != "cpu" else "cpu",
        )
        dist_helper.broadcast(t, src=0)
        val = int(t.item())
        return None if val < 0 else val

    async def backfill_range(self) -> None:
        """
        Backfill evaluations within [from_window, to_window], inclusive.
        - Skips missing/incomplete checkpoints.
        - Respects --force_reval to re-run already-evaluated windows.
        """
        assert self.from_window is not None or self.to_window is not None, (
            "backfill_range called without bounds"
        )

        # Resolve bounds (master computes, all ranks participate in broadcast)
        if self.is_master:
            resolved_from = (
                self.from_window if self.from_window is not None else self.start_window
            )
            resolved_to = self.to_window
        else:
            resolved_from = None
            resolved_to = None

        # All ranks must participate in _resolve_to_window if needed
        if self.to_window is None:
            resolved_to = await self._resolve_to_window()

        if self.is_master:
            bounds = torch.tensor(
                [
                    -1 if resolved_from is None else int(resolved_from),
                    -1 if resolved_to is None else int(resolved_to),
                ],
                dtype=torch.int64,
                device=self.device,
            )
        else:
            bounds = torch.zeros(
                2,
                dtype=torch.int64,
                device=self.device,
            )

        # Broadcast both bounds
        dist_helper.broadcast(bounds, src=0)
        resolved_from, resolved_to = [
            None if int(v.item()) < 0 else int(v.item()) for v in bounds
        ]

        if resolved_from is None or resolved_to is None:
            if self.is_master:
                tplr.logger.warning(
                    "[Master] Backfill bounds unresolved; skipping backfill."
                )
            return

        if resolved_from > resolved_to:
            # Normalize order if accidentally flipped.
            resolved_from, resolved_to = resolved_to, resolved_from

        if self.is_master:
            tplr.logger.info(
                f"[Master] Backfilling windows [{resolved_from}, {resolved_to}] for version '{self.version}'"
            )

        for w in range(resolved_from, resolved_to + 1):
            # Master checks completeness and broadcasts result
            if self.is_master:
                ready = await self.ckpt.check_checkpoint_exists(window=w)
                ready_tensor = torch.tensor(
                    [1 if ready else 0],
                    dtype=torch.int64,
                    device=self.device if self.device != "cpu" else "cpu",
                )
            else:
                # Non-master ranks prepare empty tensor for broadcast
                ready_tensor = torch.zeros(
                    1,
                    dtype=torch.int64,
                    device=self.device if self.device != "cpu" else "cpu",
                )

            # Broadcast checkpoint readiness from master to all ranks
            dist_helper.broadcast(ready_tensor, src=0)
            ready = bool(ready_tensor.item())

            if not ready:
                if self.is_master:
                    tplr.logger.info(
                        f"[Master] Skip window {w}: checkpoint not complete/available"
                    )
                continue

            await self.evaluate_window(w)
            dist_helper.safe_barrier(
                tag=f"backfill_win_{w}", local_rank=self.local_rank
            )

    async def run(self):
        """Main evaluation loop."""

        if self.is_master:
            tplr.logger.info("[Master] Starting evaluator")

        # Start background tasks
        self.comms.commitments = await self.comms.get_commitments()

        # Get the start window for global_step calculation (master fetches, then broadcasts)
        if self.is_master:
            start_window = await self.comms.get_start_window(version=self.version)
            assert start_window is not None
            self.start_window = start_window

            # Prepare tensor for broadcasting
            val = -1 if self.start_window is None else self.start_window
            tensor = torch.tensor([val], dtype=torch.long, device=self.device)
        else:
            # Non-master ranks prepare empty tensor
            tensor = torch.zeros(1, dtype=torch.long, device=self.device)

        # Broadcast start_window to all ranks
        dist_helper.broadcast(tensor, src=0)
        val = tensor.item()
        start_window = None if val == -1 else int(val)
        assert start_window is not None
        self.start_window = start_window

        # Check for initial checkpoint
        latest = await self.check_latest_checkpoint()

        # Check if bootstrap version is configured
        bootstrap_version = getattr(self.hparams, "checkpoint_init_version", None)

        if latest is None and not self.baseline_evaluated and not bootstrap_version:
            # No checkpoints and no bootstrap - run baseline once
            if self.is_master:
                tplr.logger.info(
                    "[Master] No checkpoints found and no bootstrap configured. Running baseline evaluation."
                )
            if await self.evaluate_window(start_window, is_baseline=True):
                self.baseline_evaluated = True
        elif latest is None and bootstrap_version:
            # No checkpoints but bootstrap configured - skip baseline
            if self.is_master:
                tplr.logger.info(
                    f"[Master] No checkpoints found but bootstrap version {bootstrap_version} configured. "
                    f"Skipping baseline evaluation."
                )
        elif latest is not None:
            # Evaluate latest checkpoint
            if self.is_master:
                tplr.logger.info(
                    f"[Master] Found checkpoint at window {latest}, evaluating..."
                )
            await self.evaluate_window(latest)

        # If the user specified an explicit backfill range, perform it now.
        # In backfill mode, we:
        #  - do not run baseline
        #  - evaluate all READY checkpoints in the inclusive range
        #  - exit immediately if --no_follow was passed
        if self.from_window is not None or self.to_window is not None:
            if self.is_master:
                tplr.logger.info("[Master] Entering backfill mode")
            await self.backfill_range()

            if self.no_follow:
                if self.is_master:
                    tplr.logger.info(
                        "[Master] --no_follow set; exiting after backfill."
                    )
                return

        # Main loop
        if self.is_master:
            tplr.logger.info(
                f"[Master] Entering main loop (checking every {self.config.eval_interval}s)"
            )
        loop_iteration = 0

        while True:
            loop_iteration += 1
            if self.is_master:
                tplr.logger.info(
                    f"[Master] Sleeping for {self.config.eval_interval}s (iteration {loop_iteration})"
                )
            await asyncio.sleep(cast(int, self.config.eval_interval))

            # Update state
            self.comms.commitments = await self.comms.get_commitments()

            # Check for new checkpoint
            new_window = await self.check_latest_checkpoint()

            if new_window is not None:
                if self.is_master:
                    tplr.logger.info(f"[Master] New checkpoint at window {new_window}")
                await self.evaluate_window(new_window)

                # Sync all ranks
                dist_helper.safe_barrier(
                    tag="eval_complete", local_rank=self.local_rank
                )
            else:
                if self.is_master:
                    tplr.logger.info(
                        f"[Master] No new checkpoints found. Already evaluated windows: {sorted(self.evaluated_windows)}"
                    )

    def cleanup(self):
        """Cleanup resources including active subprocesses."""
        # Terminate any active subprocess first
        if self._active_subprocess is not None:
            tplr.logger.info("Terminating active subprocess...")
            try:
                self._active_subprocess.terminate()
                try:
                    self._active_subprocess.wait(timeout=5)
                    tplr.logger.info("Subprocess terminated gracefully")
                except subprocess.TimeoutExpired:
                    tplr.logger.warning("Subprocess did not terminate, killing...")
                    self._active_subprocess.kill()
                    self._active_subprocess.wait()
                    tplr.logger.info("Subprocess killed")
            except Exception as e:
                tplr.logger.warning(f"Failed to terminate subprocess: {e}")
            self._active_subprocess = None

        if hasattr(self, "model"):
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()

        if "DATASET_BINS_PATH" in os.environ:
            del os.environ["DATASET_BINS_PATH"]


def get_config() -> bt.config:
    """Parse configuration."""
    parser = argparse.ArgumentParser(description="Templar Evaluator")

    # Core arguments
    parser.add_argument("--netuid", type=int, default=3, help="Network UID")
    parser.add_argument("--version", type=str, default=None, help="Model version")

    # Evaluation settings
    parser.add_argument(
        "--eval_interval", type=int, default=600, help="Evaluation interval (seconds)"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag,mmlu",
        help="Comma-separated evaluation tasks",
    )
    parser.add_argument(
        "--custom_eval_path", type=str, default=None, help="Custom dataset path"
    )

    # Cache settings
    parser.add_argument(
        "--cache_dir", type=str, default="models/cache", help="Model cache directory"
    )
    parser.add_argument(
        "--project", type=str, default="templar-eval", help="WandB project name"
    )

    # Backfill / control flags
    parser.add_argument(
        "--from_window",
        type=int,
        default=None,
        help="Backfill: starting window (inclusive). If omitted but --to_window is set, "
        "the start window is resolved from the network's start window.",
    )
    parser.add_argument(
        "--to_window",
        type=int,
        default=None,
        help="Backfill: ending window (inclusive). If omitted while --from_window is set, "
        "the latest READY window will be discovered remotely.",
    )
    parser.add_argument(
        "--no_follow",
        action="store_true",
        help="One-shot mode. Do not resume/tail to the latest after the initial evaluation/backfill.",
    )
    parser.add_argument(
        "--force_reval",
        action="store_true",
        help="Re-evaluate windows even if they were already evaluated in this process (ignores the in-memory set). "
        "Does not remove existing model caches.",
    )

    # Add wallet and subtensor args
    bt.subtensor.add_args(parser)

    parser.parse_args()
    config = bt.config(parser)

    return config


# Global reference for signal handler
_evaluator: Evaluator | None = None


def _signal_handler(signum, frame):
    """Handle termination signals by cleaning up evaluator."""
    global _evaluator
    sig_name = signal.Signals(signum).name
    tplr.logger.info(f"Received {sig_name}, cleaning up...")
    if _evaluator is not None:
        _evaluator.cleanup()
    sys.exit(0)


async def main():
    """Entry point."""
    global _evaluator
    config = get_config()

    # Setup logging
    if config.debug:
        tplr.debug()
    if config.trace:
        tplr.trace()

    _evaluator = Evaluator(config)
    evaluator = _evaluator

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        await evaluator.run()
    except KeyboardInterrupt:
        tplr.logger.info("Shutting down...")
    except Exception as e:
        tplr.logger.exception(f"Fatal error: {e}")
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

