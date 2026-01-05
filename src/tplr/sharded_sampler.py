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

import os
import time
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Sampler

import tplr


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _window_seed(uid: int, window: int) -> int:
    """Deterministic 32-bit seed for a (uid, window) pair."""
    return (uid * 1_000_003 ^ window) & 0xFFFF_FFFF


# ────────────────────────────────────────────────────────────────────────────
# Base class containing all shared behaviour
# ────────────────────────────────────────────────────────────────────────────
class _BaseWindowSampler(Sampler, ABC):
    """
    Common functionality for MinerSampler / EvalSampler.

    Subclasses implement `_global_indices()` which must return a 1-D
    NumPy array of **global** indices (before rank slicing).
    """

    def __init__(
        self,
        dataset: tplr.SharedShardedDataset,
        uid: int,
        window: int,
        *,
        steps_per_window: int,
        micro_bs: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        tp_degree: int | None = None,
    ):
        self._dataset_ref = dataset
        self.dataset_len = len(dataset)
        self.steps_per_window = steps_per_window
        self.micro_bs = micro_bs
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        # grad-accumulation factor (also serves as a symmetry check)
        # With TP, only count DP dimension for effective batch size
        # If tp_degree not provided, fall back to environment variable
        if tp_degree is None:
            tp_degree = int(os.environ.get("TP_DEGREE", 1))
        self.tp_degree = tp_degree
        if (
            tp_degree > 1
            and self.world_size >= tp_degree
            and self.world_size % tp_degree == 0
        ):
            effective_dp = self.world_size // tp_degree
            tplr.logger.info(
                f"[TP Sampler] rank={self.rank}, TP={tp_degree}, world_size={self.world_size}, "
                f"effective_dp={effective_dp}, using DP world size for grad_accum_steps"
            )
        else:
            effective_dp = self.world_size

        denom = micro_bs * effective_dp
        self.grad_accum_steps = batch_size // denom
        self.set_window_uid(uid, window)

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def set_window_uid(self, uid: int, window: int):
        """Update to a new (uid, window) and recompute local indices."""
        self.uid, self.window = uid, window

        global_indices = self._global_indices()

        # With TP, all ranks in the same TP group must see the SAME data
        tp_degree = self.tp_degree

        # Guard: only apply TP sharding if world_size is divisible by tp_degree
        if (
            tp_degree > 1
            and self.world_size >= tp_degree
            and self.world_size % tp_degree == 0
        ):
            # Calculate DP rank (which TP group this rank belongs to)
            dp_rank = self.rank // tp_degree
            dp_world_size = self.world_size // tp_degree

            # Shard only across DP dimension (all ranks in same TP group get same data)
            self._local = global_indices[dp_rank::dp_world_size].tolist()

        else:
            # Original behavior: shard across all ranks (for FSDP/DDP)
            # Also fallback if TP config is invalid
            if tp_degree > 1 and self.world_size % tp_degree != 0:
                tplr.logger.warning(
                    f"[TP Sampler] Invalid TP config: world_size={self.world_size} not divisible by "
                    f"TP_DEGREE={tp_degree}. Falling back to non-TP sharding."
                )
            self._local = global_indices[self.rank :: self.world_size].tolist()

    def __iter__(self):
        return iter(self._local)

    def __len__(self):
        return len(self._local)

    # ------------------------------------------------------------------ #
    # helper: map dataset indices ➜ sample_id (used for hashing receipts)
    # ------------------------------------------------------------------ #
    def ids_for_indices(self, idx_list: list[int]) -> list[int]:
        ds = self._dataset_ref
        return [ds.sample_id(i) for i in idx_list]

    # --------------------------------------------------------------------- #
    # to be implemented by subclasses
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _global_indices(self) -> np.ndarray:  # noqa: N802
        """
        Return a NumPy array of indices for **all** GPUs in the window.
        Must be deterministic w.r.t. `(uid, window)`.
        """
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────────────
# MinerSampler
# ────────────────────────────────────────────────────────────────────────────
class MinerSampler(_BaseWindowSampler):
    """
    Deterministic, rank-aware sampler for miner training.
    """

    # Explicit constructor so we can pass the dataset reference upward.
    def __init__(
        self,
        dataset: tplr.SharedShardedDataset,
        uid: int,
        window: int,
        *,
        steps_per_window: int,
        micro_bs: int,
        batch_size: int,
        target_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        tp_degree: int | None = None,
    ):
        self.target_batch_size = target_batch_size
        super().__init__(
            dataset,
            uid,
            window,
            steps_per_window=steps_per_window,
            micro_bs=micro_bs,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            tp_degree=tp_degree,
        )

    def _global_indices(self) -> np.ndarray:
        wanted = self.steps_per_window * self.target_batch_size
        if wanted > self.dataset_len:
            raise ValueError(
                f"Window needs {wanted} samples but dataset has only {self.dataset_len}"
            )

        rng = np.random.default_rng(_window_seed(self.uid, self.window))
        return rng.choice(self.dataset_len, size=wanted, replace=False)


# ────────────────────────────────────────────────────────────────────────────
# EvalSampler
# ────────────────────────────────────────────────────────────────────────────
class EvalSampler(_BaseWindowSampler):
    """
    Deterministic sampler for validators.  Reproduces the miner’s
    training pool first and then draws `validation_bs` examples from it.
    """

    _cached_indices: dict[tuple[int, int], np.ndarray] = {}

    def __init__(  # signature differs, so we must override
        self,
        dataset: tplr.SharedShardedDataset,
        uid: int,
        window: int,
        *,
        steps_per_window: int,
        micro_bs: int,
        batch_size: int,
        validation_bs: int,
        rank: int = 0,
        world_size: int = 1,
        tp_degree: int | None = None,
    ):
        if validation_bs > batch_size:
            raise ValueError("validation_bs must be ≤ batch_size")

        self.validation_bs = validation_bs
        super().__init__(
            dataset,
            uid,
            window,
            steps_per_window=steps_per_window,
            micro_bs=micro_bs,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            tp_degree=tp_degree,
        )

    def _eval_seed(self) -> int:
        """
        A non‑deterministic seed that still stays constant for one full
        validator window.

        • derive from the chain timestamp so every new window changes it
        • mix in uid so two validators do not pick the *same* random slice
        """
        now_ns = int(time.time_ns())  # 64‑bit noise source
        return (self.uid * 0x9E3779B1 ^ self.window ^ now_ns) & 0xFFFF_FFFF

    def _global_indices(self) -> np.ndarray:
        key = (self.uid, self.window)
        if key in self._cached_indices:
            return self._cached_indices[key]

        train_total = self.steps_per_window * self.batch_size
        if train_total > self.dataset_len:
            raise ValueError("Training pool larger than dataset!")

        seed = _window_seed(self.uid, self.window)

        # reconstruct miner training pool
        rng_train = np.random.default_rng(seed)
        pool = rng_train.choice(self.dataset_len, size=train_total, replace=False)

        # draw validation subset (non‑deterministic, but once‑per‑window)
        rng_val = np.random.default_rng(self._eval_seed())
        val_idx = rng_val.choice(pool, size=self.validation_bs, replace=False)

        self._cached_indices[key] = val_idx
        return val_idx
