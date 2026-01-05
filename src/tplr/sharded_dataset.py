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
import os
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import tplr


def compute_shard_state(
    global_step: int,
    outer_steps_per_shard: int,
    reset_outer_step: int | None = None,
) -> tuple[int, int]:
    """
    Compute shard epoch and index with optional reset point.

    Returns:
        (epoch, shard_index)
        epoch increments when reset_outer_step is reached; shard_index resets to 0.
    """
    if outer_steps_per_shard <= 0:
        return 0, 0

    if (
        reset_outer_step is None
        or reset_outer_step < 0
        or global_step < reset_outer_step
    ):
        return 0, global_step // outer_steps_per_shard

    adjusted_step = max(global_step - reset_outer_step, 0)
    return 1, adjusted_step // outer_steps_per_shard


class SharedShardedDataset(Dataset):
    """
    Memory-maps the *pre-processed* dataset produced by `run_preprocessing()`.

    • Zero runtime concatenation or hashing – everything is done offline.
    • All ranks slice the same mmap; no extra copies.
    • Each worker sees ⌈N / world_size⌉ samples (last worker may get fewer).
    """

    def __init__(
        self,
        shard_index: int,
        sequence_length: int,
        rank: int,
        world_size: int,
        *,
        token_dtype: npt.DTypeLike = np.uint32,  # MUST match preprocessing
        file_prefix: str = "train",  # Allow custom file prefix (e.g., "val", "eval")
    ):
        super().__init__()
        t0 = time.perf_counter()
        self.seqlen = sequence_length

        # Detect DDP context (works in single-process mode too)
        self.rank = rank
        self.world = world_size
        if self.world > 1:
            dist.barrier(device_ids=[self.rank])

        self.tokens_file, self.ids_file = self.locate_shards(
            shard_index, file_prefix=file_prefix
        )
        _ = self.check_paths([self.tokens_file, self.ids_file])
        _ = self.mmap_tokens_and_ids(token_dtype)

        # should wrap in a timer
        tplr.logger.info(
            f"[Dataset] rank {self.rank}: init done in {time.perf_counter() - t0:.1f}s "
            f"({self.total_samples} samples)"
        )

    @staticmethod
    def locate_shards(
        shard_index: int,
        custom_path: os.PathLike | None = None,
        file_prefix: str = "train",
    ) -> list[Path]:
        """Locates the file paths for a given shard index.

        Args:
            shard_index: The index of the shard to locate.
            custom_path: A custom path to search for the shards. If not provided,
                it uses the `DATASET_BINS_PATH` environment variable.

        Returns:
            A tuple containing the file paths for the tokens and IDs of the shard.

        Raises:
            ValueError: If the dataset path is not configured.
        """
        shards_path = custom_path or os.getenv("DATASET_BINS_PATH")
        if shards_path is None:
            raise ValueError(
                "Dataset path not configured. Set $DATASET_BINS_PATH or provide custom_path"
            )

        tokens_file = os.path.join(shards_path, f"{file_prefix}_{shard_index:06d}.npy")
        # For anneal mode, sample_ids files use "sample_ids_anneal_" prefix
        if file_prefix == "anneal":
            ids_file = os.path.join(
                shards_path, f"sample_ids_{file_prefix}_{shard_index:06d}.npy"
            )
        else:
            ids_file = os.path.join(shards_path, f"sample_ids_{shard_index:06d}.npy")

        return tokens_file, ids_file

    @staticmethod
    def check_paths(paths: list[os.PathLike]) -> None:
        for path in paths:
            if not os.path.exists(path):
                *dir_path, file = path.split("/")
                dir_path = "/".join(dir_path)
                raise FileNotFoundError(
                    f"Pre-processed file {file} not found in {dir_path}. "
                    "Run the preprocessing script first."
                )
        return

    def mmap_tokens_and_ids(self, token_dtype: npt.DTypeLike):
        """Memory-maps the tokens and sample IDs from their respective files.

        This method opens the token and ID files as memory-mapped arrays,
        allowing for efficient access to the data without loading it all into RAM.

        Args:
            token_dtype: The numpy data type of the tokens.
        """
        # ────────────────────────── mmap tokens & ids ───────────────────────────
        tokens_path = Path(self.tokens_file)
        ids_path = Path(self.ids_file)

        # Tokens: support .npy (NumPy format) and raw .bin
        if tokens_path.suffix == ".npy":
            # Correct way to memory-map a NumPy .npy file
            arr = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
            # Ensure dtype is uint32 (no copy if already correct)
            if arr.dtype != np.uint32:
                arr = arr.astype(np.uint32, copy=False)

            # Flatten if array is not 1D (defensive check for preprocessing bugs)
            if arr.ndim != 1:
                tplr.logger.warning(
                    f"[Dataset] Token file has wrong shape {arr.shape}, expected 1D. "
                    f"Flattening to 1D (file: {self.tokens_file})"
                )
                arr = arr.reshape(-1)

            tokens_mem = arr
        else:
            # Raw binary: assume little-endian uint32 from preprocessing
            tokens_mem = np.memmap(tokens_path, dtype=np.dtype("<u4"), mode="r")

        # IDs sidecar: support .npy (preferred) or raw .bin
        if ids_path.suffix == ".npy":
            ids_arr = np.load(ids_path, mmap_mode="r", allow_pickle=False)
            if ids_arr.dtype != np.uint64:
                ids_arr = ids_arr.astype(np.uint64, copy=False)
            ids_mem = ids_arr
        else:
            ids_mem = np.memmap(ids_path, dtype=np.dtype("<u8"), mode="r")

        # Wrap as torch tensors (zero-copy views)
        self.tokens = torch.from_numpy(tokens_mem)  # dtype: torch.uint32
        self.sample_ids = torch.from_numpy(ids_mem).to(torch.uint64)

        # Basic sanity: tokens length should be multiple of seqlen
        total_tokens = int(self.tokens.shape[0])
        if total_tokens % self.seqlen != 0:
            raise ValueError(
                f"Token file length ({total_tokens}) not divisible by seqlen ({self.seqlen}). "
                f"File: {self.tokens_file}"
            )

        # Calculate actual sample count from tokens (authoritative source)
        actual_samples_from_tokens = total_tokens // self.seqlen
        sample_ids_count = int(self.sample_ids.shape[0])

        # Use the smaller of the two to avoid out-of-bounds access
        self.total_samples = min(actual_samples_from_tokens, sample_ids_count)

        # Warn if there's a mismatch (indicates preprocessing bug)
        if actual_samples_from_tokens != sample_ids_count:
            tplr.logger.warning(
                f"[Dataset] Mismatch between tokens and sample_ids! "
                f"tokens={total_tokens} ({actual_samples_from_tokens} samples), "
                f"sample_ids={sample_ids_count} (file: {self.tokens_file}). "
                f"Using {self.total_samples} samples to avoid out-of-bounds access."
            )

        return

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        if idx >= self.total_samples:
            raise IndexError
        start = idx * self.seqlen
        end = start + self.seqlen
        return self.tokens[start:end]

    def sample_id(self, idx: int) -> int:
        """Retrieves the unique identifier for a sample at a given index.

        Args:
            idx: The index of the sample.

        Returns:
            The sample ID.
        """
        return int(self.sample_ids[idx].item())


class ShardedDatasetManager:
    """Manages the lifecycle of sharded datasets, including downloading and swapping."""

    def __init__(
        self,
        sequence_length: int,
        rank: int,
        world_size: int,
        comms: tplr.comms.Comms,
        token_dtype: npt.DTypeLike = np.uint32,
        file_prefix: str = "train",
        anneal_mode: bool = False,
    ):
        """Initializes the dataset manager.

        Args:
            sequence_length: The length of each sequence in the dataset.
            rank: The rank of the current process in distributed training.
            world_size: The total number of processes in distributed training.
            comms: An instance of `tplr.comms.Comms` for communication.
            token_dtype: The numpy data type of the tokens.
            file_prefix: The prefix for shard files (e.g., "train", "anneal").
            anneal_mode: If True, lock to single shard with no cycling.
        """
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.token_dtype = token_dtype
        self.file_prefix = file_prefix
        self.shard_index = 0
        self.anneal_mode = anneal_mode

        self.active_dataset: SharedShardedDataset | None = None
        self.upcoming_dataset: asyncio.Task | None = None

        self.comms = comms

        # should comms glob to know all file paths?
        self.max_dataset_idx = 14  # bucket_glob_files_idx

        if anneal_mode:
            tplr.logger.info(
                "[Dataset] Anneal mode: locked to single shard, no cycling"
            )

    @staticmethod
    def remap_shard_index(shard_index: int) -> int:
        """Remaps shard 7 to shard 13, keeps all other indices unchanged.

        Args:
            shard_index: The original shard index.

        Returns:
            The remapped shard index (13 if input was 7, otherwise unchanged).
        """
        return 13 if shard_index == 7 else shard_index

    def prepare_shard(self, shard_index: int) -> asyncio.Task:
        """Prepares a shard for use, downloading it if necessary.

        Args:
            shard_index: The index of the shard to prepare.

        Returns:
            An asyncio Task that completes when the download is finished
        """
        # Remap shard 7 to shard 13
        remapped_shard = self.remap_shard_index(shard_index)
        tokens_file, ids_file = SharedShardedDataset.locate_shards(
            remapped_shard, file_prefix=self.file_prefix
        )
        tplr.logger.info(
            f"Preparing shard {shard_index} (remapped to {remapped_shard}) at {tokens_file}"
        )

        if os.path.exists(tokens_file) and os.path.exists(ids_file):
            # if exist, return completed task
            print(f"Shard {shard_index} already exists on disk. Loading...")
            task = asyncio.create_task(asyncio.sleep(0))

        else:
            bucket = self.comms.get_own_bucket("dataset", "read")
            task = asyncio.create_task(
                self.download_files(bucket, tokens_file, ids_file)
            )

        return task

    async def download_files(
        self,
        bucket: tplr.schemas.Bucket,
        tokens_file: os.PathLike,
        ids_file: os.PathLike,
    ) -> asyncio.TaskGroup:
        """
        Downloads the shard and its indices

        Args:
            bucket: The (shared shard) r2 storage bucket
            tokens_file: The path to the tokens file in bucket
            ids_file: The path to the tokens file's indices in bucket
        """
        return await asyncio.gather(
            self.comms.s3_get_object(
                tokens_file,
                bucket,
                load_data=False,
            ),
            self.comms.s3_get_object(
                ids_file,
                bucket,
                load_data=False,
            ),
        )

    async def create_dataset(self, shard_index: int) -> SharedShardedDataset:
        """Creates a `SharedShardedDataset` instance for a given shard index.

        Args:
            shard_index: The index of the shard to create a dataset for.

        Returns:
            An instance of `SharedShardedDataset`.
        """
        # Remap shard 7 to shard 13
        remapped_shard = self.remap_shard_index(shard_index)

        # Only rank 0 downloads the shard, others wait
        if self.rank == 0:
            download_task = self.prepare_shard(shard_index)
            await download_task
        # Non-master ranks will just check if files exist (downloaded by rank 0)

        dataset = SharedShardedDataset(
            shard_index=remapped_shard,
            sequence_length=self.sequence_length,
            rank=self.rank,
            world_size=self.world_size,
            token_dtype=self.token_dtype,
            file_prefix=self.file_prefix,
        )
        return dataset

    async def initialize_datasets(self, current_shard_index: int) -> None:
        """Initializes the active and upcoming datasets.

        This method creates the dataset for the current shard index and starts
        preparing the next shard in the background.

        Args:
            current_shard_index: The index of the shard to make active.
        """
        # Update internal shard index to match the shard being loaded
        self.shard_index = current_shard_index

        self.active_dataset = await self.create_dataset(current_shard_index)

        # In anneal mode, don't prepare next shard (we stay on shard 0)
        if self.anneal_mode:
            self.upcoming_dataset = asyncio.create_task(asyncio.sleep(0))
            return

        next_shard = (current_shard_index + 1) % self.max_dataset_idx

        # Only rank 0 prepares the next shard to avoid duplicate downloads
        if self.rank == 0:
            self.upcoming_dataset = self.prepare_shard(next_shard)
        else:
            # Non-master ranks create a dummy completed task
            self.upcoming_dataset = asyncio.create_task(asyncio.sleep(0))
        return

    async def swap_datasets(self) -> int:
        """Swaps the active dataset with the upcoming one.

        This method waits for the upcoming dataset to be ready, makes it the
        active dataset, and starts preparing the next one. It also cleans up
        the files of the old dataset.
        """
        if self.anneal_mode:
            tplr.logger.info("[Dataset] Anneal mode: skipping shard swap")
            return self.shard_index

        self.shard_index += 1
        self.shard_index = self.shard_index % self.max_dataset_idx  # allow replay

        if self.upcoming_dataset:
            await self.upcoming_dataset

        old_dataset = self.active_dataset
        await self.initialize_datasets(self.shard_index)
        tplr.logger.info("successfully swapped datasets.")

        if old_dataset and self.rank == 0:
            filenames = ["tokens_file", "ids_file"]
            files_to_delete = [old_dataset.tokens_file, old_dataset.ids_file]
            for name, filepath in zip(filenames, files_to_delete):
                try:
                    os.remove(filepath)
                except FileNotFoundError:
                    tplr.logger.error(f"{name} file not available for deletion")

        del old_dataset

        return self.shard_index
