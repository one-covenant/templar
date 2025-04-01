"""Templar Autonomous Model Evaluator Service

This script implements an autonomous service that continuously evaluates the latest
model checkpoints using standardized benchmark tasks. It runs on a fixed interval
(default 10 minutes), downloads the latest model checkpoint, executes evaluations,
and logs results to InfluxDB.

Key Features:
    - Automatic checkpoint detection and evaluation
    - Multiple benchmark task support (arc_challenge, winogrande, etc.)
    - Distributed metrics logging
    - Resource management and cleanup
    - Service-oriented design for continuous operation

Environment Requirements:
    - Registered Bittensor wallet
    - InfluxDB API access
    - R2 Dataset access credentials

Required Environment Variables:
    R2_DATASET_ACCOUNT_ID: R2 dataset account identifier (see miner documentation)
    R2_DATASET_BUCKET_NAME: R2 storage bucket name (see miner documentation)
    R2_DATASET_READ_ACCESS_KEY_ID: R2 read access key (see miner documentation)
    R2_DATASET_READ_SECRET_ACCESS_KEY: R2 secret access key (see miner documentation)
    INFLUXDB_TOKEN: InfluxDB API token (optional, uses default if not provided)

Usage Examples:
    Basic run:
        $ uv run ./scripts/evaluator.py

    Custom configuration:
        $ uv run scripts/evaluator.py \\
            --netuid 3 \\
            --device cuda \\
            --tasks "arc_challenge,winogrande" \\
            --eval_interval 300

For additional environment setup, refer to the miner documentation:
https://github.com/tplr-ai/templar/blob/main/docs/miner.md
"""

import os
import json
import shutil
import torch
import asyncio
import argparse
import time
import tplr
import bittensor as bt

from typing import Optional, Tuple
from transformers.models.llama import LlamaForCausalLM

CHECKPOINT_DEFAULT_DIR: str = "checkpoints/"
MODEL_PATH: str = "models/eval"
DEFAULT_EVAL_INTERVAL: int = 60 * 10  # 10 mins default interval


def config() -> bt.Config:
    """
    Parse command-line arguments and return a configuration object.
    """

    parser = argparse.ArgumentParser(
        description="Evaluator script. Use --help to display options.",
        add_help=True,
    )
    # Removed wandb project argument
    parser.add_argument(
        "--netuid",
        type=int,
        default=3,
        help="Bittensor network UID.",
    )
    parser.add_argument(
        "--actual_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:7",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to save/load checkpoints",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=DEFAULT_EVAL_INTERVAL,
        help="Global steps between evaluations",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Override the wallet's UID",
    )

    bt.subtensor.add_args(parser)
    parser.parse_args()
    return bt.config(parser)


class Evaluator:
    """Templar Model Evaluator Component

    The Evaluator is responsible for automated benchmark evaluation of model checkpoints.
    It continuously monitors for new checkpoints by window number, downloads them, runs a
    comprehensive suite of language model evaluations, and logs results to both InfluxDB and W&B.

    Key Features:
        - Automatic checkpoint detection by window number
        - Multi-task model evaluation
        - Distributed metrics logging
        - Progress tracking via W&B
        - Resource cleanup and management

    Evaluation Flow:
        1. Monitor blockchain for new checkpoints by window number
        2. Download and load checkpoint directly when detected
        3. Run benchmark suite using lm-eval
        4. Parse and log results
        5. Clean up resources
        6. Wait for next checkpoint

    Attributes:
        config (bt.Config): Configuration object containing CLI arguments
        netuid (int): Network UID for the subnet
        model (LlamaForCausalLM): The language model being evaluated
        metrics_logger (MetricsLogger): Logger for InfluxDB metrics
        wandb_run: Weights & Biases run instance
        last_eval_window (int): Last evaluated window number
        last_block_number (int): Last processed block number
    """

    def __init__(self) -> None:
        self.config = config()
        if self.config.netuid is None:
            raise ValueError("No netuid provided")
        if self.config.device is None:
            raise ValueError("No device provided")
        # Use constant for default checkpoint directory.
        self.checkpoint_path: str = (
            self.config.checkpoint_path or CHECKPOINT_DEFAULT_DIR
        )
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        self.netuid = self.config.netuid
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.hparams = tplr.load_hparams()
        self.wallet = bt.wallet(config=self.config)

        # Mock for the comms class
        self.uid = 1

        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)

        self.tokenizer = self.hparams.tokenizer
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="model",
            config=self.config,
            netuid=self.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,
        )

        self.buckets = self.comms.get_all_buckets()
        self.last_eval_window = 0
        self.stop_event = asyncio.Event()
        self.last_block_number = 0

        # Initialize metrics logger with consistent patterns
        self.metrics_logger = tplr.metrics.MetricsLogger(
            prefix="E",
            uid=str(self.uid),
            config=self.config,
            role="evaluator",
            group="evaluations",
            job_type="eval",
        )

    async def update_state(self) -> None:
        """
        Refresh the metagraph and bucket information.
        """
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.buckets = self.comms.get_all_buckets()

    async def load_latest_model(self) -> Tuple[bool, dict, int, int]:
        """Load and prepare the latest model checkpoint for evaluation.

        This method:
        1. Fetches the latest checkpoint from storage
        2. Verifies checkpoint validity
        3. Loads model weights and momentum
        4. Updates internal state trackers

        Returns:
            Tuple containing:
            - success (bool): Whether loading succeeded
            - checkpoint_data (dict): Checkpoint metadata
            - checkpoint_window (int): Window number of checkpoint
            - global_step (int): Global training step
        """
        result = await self.comms.get_latest_checkpoint()
        tplr.logger.info(f"[DEBUG] get_latest_checkpoint() result: {result}")
        if not result:
            tplr.logger.error(
                f"No valid checkpoints found. Check bucket: {getattr(self.comms, 'bucket_name', 'unknown')}, "
                f"key_prefix: {self.comms.key_prefix}"
            )
            return (False, {}, 0, 0)

        checkpoint_data, _ = result
        tplr.logger.info(f"[DEBUG] Checkpoint data: {checkpoint_data}")

        checkpoint_start_window = checkpoint_data.get("start_window")
        checkpoint_current_window = checkpoint_data.get("current_window")

        if checkpoint_start_window is None or checkpoint_current_window is None:
            tplr.logger.error("Checkpoint missing start_window or current_window info")
            return (False, checkpoint_data, 0, 0)

        if checkpoint_current_window <= self.last_eval_window:
            tplr.logger.info(
                f"Checkpoint already evaluated (checkpoint window: {checkpoint_current_window}, "
                f"last evaluated: {self.last_eval_window})."
            )
            return (False, checkpoint_data, checkpoint_current_window, 0)

        tplr.logger.info(
            f"Loading model from checkpoint (window: {checkpoint_current_window})"
        )

        self.model.load_state_dict(
            {
                k: v.to(self.config.device)
                for k, v in checkpoint_data["model_state_dict"].items()
            }
        )
        self.model.to(self.config.device)  # type: ignore

        self.momentum = checkpoint_data["momentum"]

        global_step = checkpoint_current_window - checkpoint_start_window

        tplr.logger.info(
            f"Loaded checkpoint (start_window={checkpoint_start_window}, "
            f"current_window={checkpoint_current_window}, global_step={global_step})"
        )

        return (True, checkpoint_data, checkpoint_current_window, global_step)

    async def _evaluate(self) -> Optional[int]:
        """Execute benchmark evaluation on the current model.

        Workflow:
        1. Save model to temporary location
        2. Run lm-eval benchmark suite
        3. Parse results for each task
        4. Log metrics to InfluxDB and W&B
        5. Clean up temporary files

        Returns:
            Optional[int]: Global step number if successful, None on failure
        """
        self.comms.commitments = await self.comms.get_commitments()
        self.comms.update_peers_with_buckets()

        block_number = self.subtensor.get_current_block() - 1

        tplr.logger.info(f"Looking for new checkpoint (block: {block_number})")

        (
            success,
            checkpoint_data,
            checkpoint_window,
            global_step,
        ) = await self.load_latest_model()

        if not success:
            tplr.logger.info(
                f"No new checkpoint to evaluate (last evaluated window: {self.last_eval_window})"
            )
            return global_step

        tplr.logger.info(
            f"Starting benchmark run at global step {global_step} (checkpoint window: {checkpoint_window})"
        )
        os.makedirs(MODEL_PATH, exist_ok=True)
        self.model.save_pretrained(MODEL_PATH)
        self.hparams.tokenizer.save_pretrained(MODEL_PATH)

        results_dir = os.path.join(MODEL_PATH, "results")
        os.makedirs(results_dir, exist_ok=True)

        start_time = time.time()
        lm_eval_command = (
            f"lm-eval "
            f"--model hf "
            f"--model_args pretrained={MODEL_PATH},tokenizer={MODEL_PATH} "
            f"--tasks {self.config.tasks} "
            f"--device {self.config.device} "
            f"--batch_size {self.config.actual_batch_size} "
            f"--output_path {results_dir}"
        )

        tplr.logger.info(f"Running benchmark command: {lm_eval_command}")
        exit_code = os.system(lm_eval_command)
        runtime = time.time() - start_time

        self.metrics_logger.log(
            measurement="benchmark_metrics",
            tags={
                "global_step": global_step,
                "window": checkpoint_window,
                "block": block_number,
            },
            fields={
                "lm_eval_exit_code": float(exit_code),
                "benchmark_runtime_s": runtime,
            },
        )
        if exit_code != 0:
            tplr.logger.error("Benchmarking command failed")
            return global_step

        eval_results_dir = os.path.join(results_dir, "models__eval")
        if not os.path.exists(eval_results_dir):
            tplr.logger.error(f"Results directory not found: {eval_results_dir}")
            return global_step

        latest_file = max(
            [os.path.join(eval_results_dir, f) for f in os.listdir(eval_results_dir)],
            key=os.path.getctime,
        )
        with open(latest_file, "r") as f:
            results = json.load(f)

        for task_name, task_results in results["results"].items():
            metric_name = "acc_norm,none" if task_name != "winogrande" else "acc,none"
            if (metric_value := task_results.get(metric_name)) is not None:
                tplr.logger.info(f"Benchmark for {task_name}: {metric_value}")
                self.metrics_logger.log(
                    measurement="benchmark_task",
                    tags={
                        "task": task_name,
                        "global_step": global_step,
                        "block": block_number,
                        "window": checkpoint_window,
                    },
                    fields={"score": float(metric_value)},
                )

        self.metrics_logger.log(
            measurement="benchmark_summary",
            tags={
                "global_step": global_step,
                "window": checkpoint_window,
                "block": block_number,
            },
            fields={
                "num_tasks": len(results["results"]),
                "global_step": global_step,
                "block_number": block_number,
            },
        )

        shutil.rmtree(MODEL_PATH)
        torch.cuda.empty_cache()

        self.last_eval_window = checkpoint_window
        self.last_block_number = block_number

        tplr.logger.info(
            f"Successfully evaluated checkpoint (window: {checkpoint_window}, "
            f"global_step: {global_step}, block: {block_number})"
        )
        return global_step

    async def run(self) -> None:
        """Main evaluation loop.

        Continuously:
        1. Check for new checkpoints by window number and block
        2. Trigger evaluation when new checkpoint detected
        3. Handle interrupts and errors
        4. Maintain evaluation interval
        """
        try:
            self.comms.start_commitment_fetcher()
            self.comms.start_background_tasks()

            while not self.stop_event.is_set():
                await self.update_state()

                latest_block = self.subtensor.get_current_block()
                start_window = await self.comms.get_start_window()

                if (
                    latest_block > self.last_block_number
                    or start_window > self.last_eval_window
                ):
                    tplr.logger.info(
                        f"New checkpoint detected (block: {latest_block}, window: {start_window}), executing benchmark..."
                    )
                    await self._evaluate()
                else:
                    tplr.logger.info(
                        f"No new checkpoint available (block: {latest_block}/{self.last_block_number}, "
                        f"window: {start_window}/{self.last_eval_window})"
                    )
                await asyncio.sleep(self.config.eval_interval)  # type: ignore
        except KeyboardInterrupt:
            tplr.logger.info("Benchmark run interrupted by user")
            self.stop_event.set()
        except Exception as e:
            tplr.logger.error(f"Benchmark run failed: {e}")

    def cleanup(self) -> None:
        """
        Cleanup resources before exit.
        """
        self.stop_event.set()


def main() -> None:
    """
    Entry point for the evaluator.
    """
    evaluator = Evaluator()
    try:
        asyncio.run(evaluator.run())
    except Exception as e:
        tplr.logger.error(f"Evaluator terminated with error: {e}")
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
