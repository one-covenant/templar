# type: ignore

# Standard library imports
import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Set

# Third-party imports
import boto3
import numpy as np
import torch
import torch.nn.functional as F
from botocore.config import Config
from transformers import LlamaForCausalLM

# Local imports
import tplr
from tplr.compress import unpack_12bit_indices
from tplr.config import BUCKET_SECRETS

# GPU optimizations.
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Analyzer:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Analyzer script")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--trace", action="store_true", help="Enable trace mode")
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project name"
        )
        parser.add_argument(
            "--analysis_interval",
            type=int,
            default=60,
            help="Interval between analyses in seconds",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Device to use for computation (cuda/cpu)",
        )
        config = parser.parse_args()
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        return config

    def __init__(self):
        try:
            self.config = Analyzer.config()
            self.temp_dir = "/tmp/analyzer"
            self.processed_files_path = Path(self.temp_dir) / "processed_files.json"
            os.makedirs(self.temp_dir, exist_ok=True)

            # Load previously processed files
            self.processed_files: Set[str] = self.load_processed_files()

            self.hparams = tplr.load_hparams()

            tplr.logger.info(f"Initializing model on device: {self.config.device}")
            # Initialize model for transformer shapes
            self.model = LlamaForCausalLM(self.hparams.model_config)
            self.model.to(self.config.device)

            # Initialize compression components
            self.transformer = tplr.compress.ChunkingTransformer(
                self.model,
                target_chunk=self.hparams.target_chunk,
            )
            self.compressor = tplr.compress.TopKCompressor(
                use_quantization=True,
                quantization_bins=self.hparams.quantization_bins,
                quantization_range=self.hparams.quantization_range,
                b_choices=self.hparams.compression_b_choices,
            )

            # Initialize shapes for each parameter (like in miner/validator)
            self.xshapes = {}
            self.totalks = {}
            for n, p in self.model.named_parameters():
                transformed = self.transformer.encode(p)
                self.xshapes[n] = transformed.shape
                self.totalks[n] = transformed.numel()

            # Initialize WandB
            self.wandb = tplr.initialize_wandb(
                run_prefix="A",  # 'A' for Analyzer
                uid=0,
                config=self.config,
                group="analyzer",
                job_type="analysis",
            )

            # Initialize R2 client
            self.bucket_info = BUCKET_SECRETS["gradients"]
            self.r2_endpoint = (
                f"https://{self.bucket_info['account_id']}.r2.cloudflarestorage.com"
            )
            self.bucket_name = self.bucket_info["name"]
            self.access_key_id = self.bucket_info["credentials"]["read"][
                "access_key_id"
            ]
            self.secret_access_key = self.bucket_info["credentials"]["read"][
                "secret_access_key"
            ]

            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.r2_endpoint,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=Config(signature_version="s3v4", max_pool_connections=256),
            )

            # Initialize peer gradients storage
            self.peer_gradients = {}
            self.current_step = 0

        except Exception as e:
            tplr.logger.error(f"Error initializing Analyzer: {e}")

    def load_processed_files(self) -> Set[str]:
        """Load the set of processed files from disk."""
        if self.processed_files_path.exists():
            try:
                with open(self.processed_files_path, "r") as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                tplr.logger.warning("Corrupted processed files cache, starting fresh")
                return set()
        return set()

    def save_processed_files(self):
        """Save the set of processed files to disk."""
        with open(self.processed_files_path, "w") as f:
            json.dump(list(self.processed_files), f)

    async def run(self):
        while True:
            try:
                await self.analyze_gradients()
            except Exception as e:
                tplr.logger.error(f"Error in analysis loop: {e}")

            await asyncio.sleep(self.config.analysis_interval)

    async def analyze_gradients(self):
        """List and analyze new gradient files."""
        try:
            # List all objects in the bucket with the gathers prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            prefix = f"gathers/{tplr.__version__}/"

            new_files = 0
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if "Contents" not in page:
                    continue

                files_to_process = [
                    obj["Key"]
                    for obj in page["Contents"]
                    if obj["Key"] not in self.processed_files
                ]

                if not files_to_process:
                    continue

                new_files += len(files_to_process)
                tasks = [self.process_gradient_file(key) for key in files_to_process]
                await asyncio.gather(*tasks)

            if new_files > 0:
                tplr.logger.info(f"Processed {new_files} new gradient files")
                self.save_processed_files()

        except Exception as e:
            tplr.logger.error(f"Error listing gradient files: {e}")

    async def process_gradient_file(self, key: str):
        """Process a single gradient file with peer comparison."""
        local_file = os.path.join(self.temp_dir, f"temp_{hash(key)}.npz")

        try:
            # Extract UID and window from the key
            parts = key.split("/")
            if len(parts) < 4:  # Verify key format
                tplr.logger.warning(f"Invalid key format: {key}")
                return

            uid = int(parts[2])
            window = int(parts[3])

            # Download and load the file
            await self.download_file(key, local_file)
            data = np.load(local_file, allow_pickle=True)
            state_dict = data["state_dict"].item()
            metadata = data["metadata"].item()

            # global_step = int(metadata.get("global_step", -1))
            timestamp = metadata.get("timestamp", time.time())

            # Store gradients for comparison
            self.peer_gradients[uid] = {
                "state_dict": state_dict,
                "window": window,
                "timestamp": timestamp,
            }

            # Get all peers from same window for comparison
            window_peers = {
                peer_uid: data
                for peer_uid, data in self.peer_gradients.items()
                if data["window"] == window
            }

            if len(window_peers) > 1:
                # Compute cosine similarity matrix
                uids = sorted(window_peers.keys())
                sim_matrix = torch.zeros((len(uids), len(uids)))

                for i, uid1 in enumerate(uids):
                    grad1 = self.decode_gradient(window_peers[uid1]["state_dict"])
                    for j, uid2 in enumerate(uids[i:], i):
                        grad2 = self.decode_gradient(window_peers[uid2]["state_dict"])
                        cos_sim = F.cosine_similarity(grad1, grad2, dim=0)
                        sim_matrix[i, j] = sim_matrix[j, i] = cos_sim

                # Calculate average similarity excluding self
                avg_similarities = {}
                for i, uid1 in enumerate(uids):
                    similarities = sim_matrix[i].clone()
                    similarities[i] = 0  # Exclude self-similarity
                    avg_similarities[uid1] = similarities.sum() / (len(uids) - 1)

                # Log similarity metrics
                self.wandb.log(
                    {
                        f"analyzer/similarity/{uid}/avg_peer_similarity": avg_similarities[
                            uid
                        ],
                        f"analyzer/similarity/{uid}/max_peer_similarity": sim_matrix[
                            uids.index(uid)
                        ].max(),
                        f"analyzer/similarity/{uid}/min_peer_similarity": sim_matrix[
                            uids.index(uid)
                        ].min(),
                    },
                    step=self.current_step,
                )

            # Analyze this peer's gradients
            metrics = self.analyze_state_dict(
                state_dict, self.transformer, self.compressor
            )

            # Log metrics
            self.wandb.log(
                {
                    f"analyzer/gradients/{uid}/norm": metrics["gradient_norm"],
                    f"analyzer/indices/{uid}/reuse_ratio": metrics["index_reuse_ratio"],
                    f"analyzer/indices/{uid}/unique_count": metrics[
                        "unique_indices_count"
                    ],
                    f"analyzer/indices/{uid}/total_count": metrics[
                        "total_indices_count"
                    ],
                    f"analyzer/metadata/{uid}/window": window,
                    f"analyzer/metadata/{uid}/timestamp": timestamp,
                },
                step=self.current_step,
            )

            # Cleanup old window data
            self.cleanup_old_windows()

            # Mark as processed
            self.processed_files.add(key)
            self.current_step += 1

        except Exception as e:
            tplr.logger.error(f"Error processing gradient file {key}: {e}")
        finally:
            if os.path.exists(local_file):
                os.remove(local_file)

    async def download_file(self, key: str, local_file: str):
        """Download a file from R2 storage."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.s3_client.download_file, self.bucket_name, key, local_file
        )

    def analyze_state_dict(self, state_dict, transformer, compressor):
        """Analyze gradient statistics from state dict with peer comparisons."""
        peer_grads = {}
        peer_indices = {}

        # Process each parameter's gradients
        for n, param_data in state_dict.items():
            if n.endswith("vals"):
                param_name = n[:-4]  # Remove 'vals' suffix
                idxs = state_dict.get(f"{param_name}idxs")
                vals = state_dict.get(f"{param_name}vals")

                if idxs is not None and vals is not None:
                    # Handle 12-bit packed format (uint8 tensor)
                    if isinstance(idxs, np.ndarray):
                        idxs = torch.from_numpy(idxs).to(self.config.device)
                    else:
                        idxs = idxs.to(self.config.device)

                    # Unpack 12-bit indices using vals shape
                    idxs_unpacked = unpack_12bit_indices(idxs, vals.shape)
                    # Store unpacked indices for analysis
                    peer_indices[param_name] = set(
                        idxs_unpacked.flatten().cpu().numpy()
                    )

                    if isinstance(vals, np.ndarray):
                        vals = torch.from_numpy(vals).to(self.config.device)

                    # Decode the gradient using transformer/compressor
                    decoded_grad = transformer.decode(
                        compressor.decompress(
                            torch.zeros_like(
                                transformer.encode(
                                    dict(self.model.named_parameters())[param_name]
                                )
                            ),
                            idxs,
                            vals,
                            self.xshapes[param_name],
                            self.totalks[param_name],
                        )
                    )
                    peer_grads[param_name] = decoded_grad.flatten()

        if not peer_grads:
            return {}

        # Concatenate all parameter gradients
        all_grads = torch.cat([grad for grad in peer_grads.values()])

        # Calculate gradient norm
        grad_norm = float(torch.norm(all_grads).item())

        # Analyze index patterns
        total_indices = sum(len(indices) for indices in peer_indices.values())
        unique_indices = sum(len(set(indices)) for indices in peer_indices.values())
        index_reuse_ratio = unique_indices / total_indices if total_indices > 0 else 0

        return {
            "gradient_norm": grad_norm,
            "index_reuse_ratio": index_reuse_ratio,
            "unique_indices_count": unique_indices,
            "total_indices_count": total_indices,
        }

    def decode_gradient(self, state_dict):
        """Decode gradient from state dict."""
        decoded_grads = []
        for n, param_data in state_dict.items():
            if n.endswith("vals"):
                param_name = n[:-4]
                idxs = state_dict.get(f"{param_name}idxs")
                vals = state_dict.get(f"{param_name}vals")
                if idxs is not None and vals is not None:
                    # Handle 12-bit packed format (uint8 tensor)
                    if isinstance(idxs, np.ndarray):
                        idxs = torch.from_numpy(idxs).to(self.config.device)
                    else:
                        idxs = idxs.to(self.config.device)

                    if isinstance(vals, np.ndarray):
                        vals = torch.from_numpy(vals).to(self.config.device)

                    decoded = self.transformer.decode(
                        self.compressor.decompress(
                            torch.zeros_like(
                                self.transformer.encode(
                                    dict(self.model.named_parameters())[param_name]
                                )
                            ),
                            idxs,
                            vals,
                            self.xshapes[param_name],
                            self.totalks[param_name],
                        )
                    )
                    decoded_grads.append(decoded.flatten())
        return torch.cat(decoded_grads) if decoded_grads else torch.tensor([])

    def cleanup_old_windows(self, window_retention=5):
        """Cleanup gradient data from old windows."""
        if hasattr(self, "peer_gradients"):
            current_windows = {data["window"] for data in self.peer_gradients.values()}
            if len(current_windows) > window_retention:
                min_window_to_keep = sorted(current_windows)[-window_retention]
                self.peer_gradients = {
                    uid: data
                    for uid, data in self.peer_gradients.items()
                    if data["window"] >= min_window_to_keep
                }


def main():
    analyzer = Analyzer()
    asyncio.run(analyzer.run())


if __name__ == "__main__":
    main()
