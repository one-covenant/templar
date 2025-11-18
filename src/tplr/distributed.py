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

"""
Distributed training utilities for multi-GPU training.
"""

import os
import time
from contextlib import nullcontext
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor as DT

import tplr


class DistributedHelper:
    """Helper class for distributed training operations."""

    def __init__(self):
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.is_master = self.rank == 0
        self.device = None

    def init_process_group(
        self, backend: str = "nccl", timeout_minutes: int = 45
    ) -> None:
        """Initialize the distributed process group."""
        if not dist.is_initialized() and self.world_size > 1:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
                rank=self.rank,
                world_size=self.world_size,
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")

            tplr.logger.info(
                f"[Distributed] Initialized process group - rank={self.rank}, "
                f"world_size={self.world_size}, local_rank={self.local_rank}, "
                f"master={self.is_master}, device={self.device}"
            )

    def destroy_process_group(self) -> None:
        """Cleanup distributed process group."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def is_distributed(self) -> bool:
        """Check if distributed training is enabled."""
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def should_continue(self, local_has_batch: bool, device: torch.device) -> bool:
        """
        Synchronize across all ranks. If *any* rank runs out of batches, all must stop.

        Args:
            local_has_batch: Whether this rank has more batches
            device: Device to use for synchronization

        Returns:
            True if all ranks should continue, False otherwise
        """
        if not self.is_distributed():
            return local_has_batch

        flag_tensor = torch.tensor([int(local_has_batch)], device=device)
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
        return bool(flag_tensor.item())

    def ddp_reduce(
        self,
        value: int | float | torch.Tensor,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
        device: torch.device | None = None,
    ) -> float:
        """
        Reduce value across all ranks and return a python float.

        Args:
            value: Value to reduce
            op: Reduction operation (SUM, AVG, MIN, MAX)
            device: Device to use for reduction

        Returns:
            Reduced value as a float
        """
        # Single-GPU fast path
        if not self.is_distributed():
            return float(value.item() if isinstance(value, torch.Tensor) else value)

        # Use provided device or default
        if device is None:
            device = self.device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Convert to tensor on the right device
        if not isinstance(value, torch.Tensor):
            tensor = torch.tensor(float(value), device=device)
        else:
            tensor = value.to(device)

        dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def barrier(self, device_ids: list | None = None) -> None:
        """
        Synchronization barrier across all ranks.

        Args:
            device_ids: Optional device IDs for NCCL backend
        """
        if not self.is_distributed():
            return

        if device_ids is not None and dist.get_backend() == "nccl":
            dist.barrier(device_ids=device_ids)
        else:
            dist.barrier()

    def safe_barrier(self, tag: str = "", local_rank: int | None = None) -> bool:
        """
        Safe barrier with fallback handling for older PyTorch versions.

        Args:
            tag: Tag for logging
            local_rank: Local rank for device-specific barrier

        Returns:
            True if barrier succeeded, False otherwise
        """
        if not self.is_distributed():
            return True

        try:
            # If we're on NCCL, ensure the barrier runs on this rank's CUDA device.
            # Fall back cleanly on older PyTorch or non-NCCL backends.
            if torch.cuda.is_available() and local_rank is not None:
                try:
                    dist.barrier(device_ids=[local_rank])
                    return True
                except TypeError:
                    # Older PyTorch or backend that doesn't accept device_ids
                    pass
                except Exception as e:
                    # Unexpected error with device_ids; try a plain barrier
                    if tag:
                        tplr.logger.warning(
                            f"[barrier:{tag}] device_ids barrier warn: {e}"
                        )
            dist.barrier()
            return True
        except Exception as e:
            if tag:
                tplr.logger.error(f"[barrier:{tag}] failed: {e}")
            return False

    def all_ok(self, ok: bool, device: torch.device, tag: str = "") -> bool:
        """
        Group-wide MIN-reduce on a 1/0 flag: returns True only if all ranks reported ok=True.
        Never raises; logs and returns False if collectives fail.

        Args:
            ok: Local flag value
            device: Device for synchronization
            tag: Optional tag for logging

        Returns:
            True if all ranks reported ok=True
        """
        if not self.is_distributed():
            return ok

        try:
            t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            out = bool(t.item())
            if not out and self.is_master and tag:
                tplr.logger.warning(f"[sync:{tag}] some rank failed; skipping step")
            return out
        except Exception as e:
            if tag:
                tplr.logger.error(f"[sync:{tag}] all_reduce failed: {e}")
            return False

    def any_ok(self, ok: bool, device: torch.device, tag: str = "") -> bool:
        """
        Group-wide MAX-reduce on a 1/0 flag: returns True if any rank reported ok=True.
        Never raises; logs and returns False if collectives fail.

        Args:
            ok: Local flag value
            device: Device for synchronization
            tag: Optional tag for logging

        Returns:
            True if any rank reported ok=True
        """
        if not self.is_distributed():
            return ok

        try:
            t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            out = bool(t.item())
            if out and self.is_master and tag:
                tplr.logger.info(f"[sync:{tag}] at least one rank reported True")
            return out
        except Exception as e:
            if tag:
                tplr.logger.error(f"[sync:{tag}] all_reduce failed: {e}")
            return False

    def all_agree(self, agree: bool, device: torch.device, tag: str = "") -> bool:
        """
        Group-wide AND operation: returns True only if ALL ranks reported agree=True.
        If any rank reports agree=False, returns False.
        Never raises; logs and returns the local value if collectives fail.

        Args:
            agree: Local flag value (True if this rank agrees with the condition)
            device: Device for synchronization
            tag: Optional tag for logging

        Returns:
            True only if ALL ranks reported agree=True
        """
        if not self.is_distributed():
            return agree

        try:
            # Use MIN: if agree=True, we send 1; if agree=False, we send 0
            # After MIN, if result is 1, all ranks agreed (all sent 1)
            # If result is 0, at least one rank disagreed (sent 0)
            t = torch.tensor([1 if agree else 0], dtype=torch.int32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            out = t.item() == 1  # True only if all ranks agreed
            if not out and self.is_master and tag:
                tplr.logger.debug(f"[sync:{tag}] at least one rank disagreed")
            return out
        except Exception as e:
            if tag:
                tplr.logger.error(f"[sync:{tag}] all_reduce failed: {e}")
            return agree  # Return local value on failure

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        """
        Broadcast tensor from source rank to all other ranks.

        Args:
            tensor: Tensor to broadcast (modified in-place)
            src: Source rank for broadcast
        """
        if self.is_distributed():
            dist.broadcast(tensor, src=src)

    def gather_object(
        self, obj: Any, object_list: list | None = None, dst: int = 0
    ) -> list | None:
        """
        Gather arbitrary Python objects from all ranks to destination rank.

        Args:
            obj: Object to send from this rank
            object_list: List to receive gathered objects (only needed on dst rank)
            dst: Destination rank

        Returns:
            List of gathered objects on dst rank, None on other ranks
        """
        if not self.is_distributed():
            return [obj] if self.rank == dst else None

        if self.rank == dst and object_list is None:
            object_list = [None] * self.world_size

        dist.gather_object(obj, object_list, dst=dst)
        return object_list if self.rank == dst else None

    def all_gather_object(self, obj: Any, object_list: list | None = None) -> list:
        """
        Gather arbitrary Python objects from all ranks to all ranks.

        Args:
            obj: Object to send from this rank
            object_list: List to receive gathered objects (will be created if None)

        Returns:
            List of gathered objects from all ranks
        """
        if not self.is_distributed():
            return [obj]

        if object_list is None:
            object_list = [None] * self.world_size

        dist.all_gather_object(object_list, obj)
        return object_list

    def all_reduce_flag(self, ok: bool, device: torch.device, tag: str = "") -> bool:
        """
        Group-wide MIN-reduce on a 1/0 flag: returns True only if all ranks reported ok=True.

        Args:
            ok: Local flag value
            device: Device for synchronization
            tag: Optional tag for logging

        Returns:
            True if all ranks reported ok=True
        """
        if not self.is_distributed():
            return ok

        t = torch.tensor([int(ok)], dtype=torch.uint8, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        result = bool(t.item())

        if not result and self.is_master and tag:
            tplr.logger.warning(f"[sync:{tag}] some rank failed; skipping step")

        return result

    def batched_all_reduce(
        self,
        values: dict[str, float | int],
        device: torch.device,
        ops: dict[str, dist.ReduceOp.RedOpType] | None = None,
    ) -> dict[str, float]:
        """
        Perform batched all-reduce on multiple scalar values in a single operation.
        This is much more efficient than multiple individual all-reduce calls.

        Args:
            values: Dictionary of {name: value} to reduce
            device: Device to perform reduction on
            ops: Optional dictionary of {name: ReduceOp} for per-value operations.
                 Defaults to SUM for all values.

        Returns:
            Dictionary of {name: reduced_value}
        """
        if not self.is_distributed():
            return {k: float(v) for k, v in values.items()}

        if not values:
            return {}

        # Default to SUM for all operations
        if ops is None:
            ops = {}

        # Group values by operation type to minimize all-reduce calls
        op_groups: dict[dist.ReduceOp.RedOpType, dict[str, float]] = {}
        for name, value in values.items():
            op = ops.get(name, dist.ReduceOp.SUM)
            if op not in op_groups:
                op_groups[op] = {}
            op_groups[op][name] = float(value)

        # Perform one all-reduce per operation type
        results = {}
        for op, group_values in op_groups.items():
            # Pack all values for this operation into a single tensor
            names = list(group_values.keys())
            tensor = torch.tensor(
                [group_values[name] for name in names],
                dtype=torch.float32,
                device=device,
            )

            # Single all-reduce for all values with this operation
            dist.all_reduce(tensor, op=op)

            # Unpack results
            for i, name in enumerate(names):
                results[name] = float(tensor[i].item())

        return results

    def _model_device(self, model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    def _get_offload_stream(self, model: torch.nn.Module):
        if not torch.cuda.is_available():
            return None
        stream = getattr(model, "_offload_stream", None)
        if stream is None:
            # One dedicated stream per model instance, *on the model's device*
            stream = torch.cuda.Stream(device=self._model_device(model))
            setattr(model, "_offload_stream", stream)
        return stream

    def get_offloaded_params(self, model: torch.nn.Module) -> tuple:
        """
        Snapshot current parameters into *reusable pinned CPU buffers* with async D2H copies.

        Returns
        -------
        (params_offloaded, param_specs)
            params_offloaded: list of CPU tensors (pinned) that hold the saved local shard
            param_specs: metadata for DTensor recreation at restore time
        """
        params_offloaded = []
        param_info = []

        # Use a dedicated stream to let D2H issue asynchronously; we fence before returning.
        stream = self._get_offload_stream(model)

        t0 = time.time()
        # Launch all copies on the offload stream
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())

        with torch.inference_mode():
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                for p in model.parameters():
                    # Source (local shard or regular tensor)
                    if isinstance(p, DT):
                        src_local = p.to_local()
                        cpu_buf = getattr(p, "_cpu_offload_buf", None)
                        if (
                            cpu_buf is None
                            or cpu_buf.shape != src_local.shape
                            or cpu_buf.dtype != src_local.dtype
                            or cpu_buf.device.type != "cpu"
                        ):
                            # Reuse a single pinned buffer per param across windows
                            cpu_buf = torch.empty_like(
                                src_local, device="cpu", pin_memory=True
                            )
                            setattr(p, "_cpu_offload_buf", cpu_buf)
                        cpu_buf.copy_(
                            src_local, non_blocking=True
                        )  # async D2H into pinned memory
                        params_offloaded.append(cpu_buf)
                        param_info.append(
                            {
                                "is_dtensor": True,
                                "device_mesh": p.device_mesh,
                                "placements": p.placements,
                                # (local shape kept for sanity)
                                "local_shape": src_local.shape,
                                "dtype": str(src_local.dtype).replace("torch.", ""),
                            }
                        )
                    else:
                        # Regular (non-DTensor) parameter
                        src = (
                            p if isinstance(p, torch.Tensor) else p.data
                        )  # Parameter is Tensor subclass
                        cpu_buf = getattr(p, "_cpu_offload_buf", None)
                        if (
                            cpu_buf is None
                            or cpu_buf.shape != src.shape
                            or cpu_buf.dtype != src.dtype
                            or cpu_buf.device.type != "cpu"
                        ):
                            cpu_buf = torch.empty_like(
                                src, device="cpu", pin_memory=True
                            )
                            setattr(p, "_cpu_offload_buf", cpu_buf)
                        cpu_buf.copy_(src, non_blocking=True)
                        params_offloaded.append(cpu_buf)
                        param_info.append(
                            {
                                "is_dtensor": False,
                                "dtype": str(src.dtype).replace("torch.", ""),
                            }
                        )

        # Fence: ensure snapshot is complete and visible
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
            stream.synchronize()

        tplr.logger.info(
            f"[ParamOffload] snap {len(params_offloaded)} params in {time.time() - t0:.3f}s"
        )

        return params_offloaded, param_info

    def restore_offloaded_params(
        self,
        model: torch.nn.Module,
        params_offloaded: list,
        param_specs: list,
    ) -> None:
        """
        Efficiently compute param deltas and restore weights without extra GPU temporaries.

        For each parameter p and saved CPU buffer B:
          1) Put B into the *grad storage* (local GPU tensor or DTensor), non_blocking.
          2) Compute Δ = B - p  in-place (Δ lives where grad lives).
          3) Restore params by   p += Δ   (which is equivalent to p ← B).
          4) Leave p.grad = Δ   for your compression stage.

        For DTensors, p.grad is created as a DTensor over the same mesh/placements
        (so your GFULL path in prepare_gradient_dict keeps working).
        """
        stream = self._get_offload_stream(model)

        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())

        t0 = time.time()
        with torch.no_grad():
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                it = zip(zip(params_offloaded, param_specs), model.parameters())
                for (saved_cpu, meta), p in it:
                    if isinstance(p, DT) and meta.get("is_dtensor", False):
                        # Stage into p.grad's *own* storage so it is freed by zero_grad(set_to_none=True)
                        local = p.to_local()
                        # If p.grad is already a DTensor with matching sharding, reuse its local storage.
                        grad_dt: DT | None = None
                        if isinstance(p.grad, DT):
                            try:
                                g_loc = p.grad.to_local()
                                if (
                                    g_loc.shape == local.shape
                                    and g_loc.dtype == local.dtype
                                    and g_loc.device == local.device
                                ):
                                    # Reuse existing grad local buffer
                                    g_loc.copy_(saved_cpu, non_blocking=True)
                                    grad_dt = p.grad  # reuse wrapper
                                else:
                                    p.grad = None  # drop mismatched buffer
                            except Exception:
                                p.grad = None
                        if grad_dt is None:
                            # Fresh local staging owned by p.grad
                            g_loc = torch.empty_like(local, device=local.device)
                            g_loc.copy_(saved_cpu, non_blocking=True)
                            grad_dt = DT.from_local(
                                g_loc,
                                device_mesh=meta["device_mesh"],
                                placements=meta["placements"],
                                run_check=False,
                            )
                            p.grad = grad_dt

                        # Δ = saved - current (in-place on p.grad's storage)
                        grad_dt.sub_(p)
                        # p ← p + Δ  (now equals saved)
                        p.data.add_(grad_dt)
                    else:
                        # Non-DTensor path
                        tensor = p if isinstance(p, torch.Tensor) else p.data
                        # Ensure we have a GPU grad buffer to stage 'saved'
                        if (
                            p.grad is None
                            or p.grad.shape != tensor.shape
                            or p.grad.device != tensor.device
                        ):
                            p.grad = torch.empty_like(tensor, device=tensor.device)

                        # H2D copy saved -> p.grad (non_blocking)
                        p.grad.copy_(saved_cpu, non_blocking=True)

                        # Δ = saved - current (in-place on p.grad)
                        p.grad.sub_(tensor)

                        # Restore params: p = p + Δ  (== saved)
                        tensor.add_(p.grad)

        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
            stream.synchronize()

        tplr.logger.info(
            f"[ParamRestore] restored {len(params_offloaded)} params in {time.time() - t0:.3f}s"
        )


# Global instance for convenience
dist_helper = DistributedHelper()
