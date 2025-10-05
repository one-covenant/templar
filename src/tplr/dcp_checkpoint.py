# Copyright (c) 2025 tplr.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import asyncio
import json
import re
import shutil
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from rich.markup import escape as _escape
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.state_dict import (
    ValueType,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    async_save,
    save,
)
from torch.distributed.checkpoint.stateful import Stateful

import tplr


# Safe escaping for untrusted log inserts (paths, keys, exceptions, etc.)
def _safe(obj) -> str:
    return _escape(str(obj))


# ── Model-only Stateful (Titan-compatible distributed state dicts) ─────────────
class AppState(Stateful):
    def __init__(self, model):
        self.model = model

    def state_dict(self) -> dict[str, ValueType]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state: dict[str, ValueType]) -> None:
        set_model_state_dict(self.model, state)


class SnapshotState(Stateful):
    """Immutable snapshot of a model's distributed state dict (CPU-offloaded)."""

    def __init__(self, model):
        # Offload shards to CPU to decouple from in-flight training updates.
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        self._snap = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )

    def state_dict(self) -> dict[str, ValueType]:
        return self._snap

    # load_state_dict is never called for saving; keep a no-op to satisfy interface.
    def load_state_dict(self, state: dict[str, ValueType]) -> None:
        return


# ── Utils ─────────────────────────────────────────────────────────────────────
def _rank(group: dist.ProcessGroup | None = None) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank(group) if group is not None else dist.get_rank()


def _world(group: dist.ProcessGroup | None = None) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group) if group is not None else dist.get_world_size()


def _is_meta(p: str | Path) -> bool:
    """True if a filename/key refers to metadata (DCP .metadata or *.json sidecars)."""
    n = Path(p).name.lower()
    return n == ".metadata" or n.endswith(".metadata") or n.endswith(".json")


def _barrier(group: dist.ProcessGroup | None = None) -> None:
    if dist.is_available() and dist.is_initialized():
        # Only provide device_ids for default/NCCL group to avoid backend mismatch.
        if torch.cuda.is_available() and group is None:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier(group=group)


def _mb(b: int) -> float:
    return b / (1024 * 1024) if b else 0.0


_OWNER_RE = re.compile(r"__([0-9]+)_[0-9]+\.distcp$")


def _owner_rank_from_name(name: str) -> int | None:
    m = _OWNER_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass(slots=True)
class Layout:
    version: str
    window: int | None = None

    @property
    def prefix(self) -> str:
        base = f"checkpoints/{self.version}"
        return f"{base}/{self.window}" if self.window is not None else base


class DCPCheckpointer:
    """
    TorchTitan‑friendly DCP checkpoint manager.

    • All ranks in the saving PG call save_local()  → sharded checkpoint to repo_root/checkpoints/…
    • Per‑rank upload split: each rank uploads its shard subset; rank‑0 uploads metadata + _LATEST.json
    • Optional background upload; delete local after success
    • Download + DCP load (reshards to miner topology)
    """

    def __init__(
        self,
        comms: "tplr.Comms",
        *,
        uid: int,
        version: str,
        repo_root: str | Path = ".",
    ):
        self.comms = comms
        self.uid = int(uid)
        self.version = version
        self.repo_root = Path(repo_root).resolve()
        self._bg: set[asyncio.Task] = set()

        # Optional CPU process group for async_save; created lazily.
        self._cpu_pg: dist.ProcessGroup | None = None
        if dist.is_available() and dist.is_initialized():
            try:
                self._cpu_pg = dist.new_group(
                    ranks=list(range(_world())), backend=dist.Backend.GLOO
                )
                tplr.logger.info(
                    f"[DCP] created CPU(Gloo) process group for async_save on rank {_rank()}"
                )
            except Exception as e:
                tplr.logger.warning(f"[DCP] failed to create CPU(Gloo) PG: {_safe(e)}")

    # ── Paths ──────────────────────────────────────────────────────────────────
    def _local_dir(self, layout: Layout) -> Path:
        d = self.repo_root / layout.prefix
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Save (async local first; Titan-style if available) ─────────────────────
    async def save_local(
        self,
        *,
        model,
        window: int,
        sync_window: int,
        topology: str = "FSDP",
        process_group: dist.ProcessGroup | None = None,  # HSDP: pass shard PG
    ) -> Path:
        """
        Call on ALL ranks in the saving PG (global PG for FSDP/TP/DP; one shard_group PG for HSDP).
        Produces sharded files + .metadata under repo_root/checkpoints/<version>/<window>.
        """
        layout = Layout(self.version, window)
        out_dir = self._local_dir(layout)
        state = {"app": AppState(model)}

        t0 = time.perf_counter()
        tplr.logger.info(
            f"[DCP][save] rank {_rank(process_group)}/{_world(process_group)} → begin local save "
            f"(window={window}, dir={_safe(out_dir)})"
        )
        save(
            state_dict=state,
            checkpoint_id=str(out_dir),
            process_group=process_group,
        )
        dt = time.perf_counter() - t0

        if _rank(process_group) == 0:
            sidecar = {
                "version": self.version,
                "window": int(window),
                "sync_window": int(sync_window),
                "world_size_at_save": int(_world(process_group)),
                "topology": topology,
                "uid": self.uid,
            }
            (out_dir / "extra_metadata.json").write_text(json.dumps(sidecar, indent=2))

        _barrier(process_group)

        # Post‑barrier: count files/bytes that exist on disk (best‑effort)
        try:
            files = [p for p in out_dir.iterdir() if p.is_file()]
            total_bytes = sum(p.stat().st_size for p in files)
            tplr.logger.info(
                f"[DCP][save] rank {_rank(process_group)}/{_world(process_group)} ← done in {dt:.2f}s "
                f"(~{len(files)} files, ~{_mb(total_bytes):.2f} MiB)"
            )
        except Exception:
            tplr.logger.info(
                f"[DCP][save] rank {_rank(process_group)}/{_world(process_group)} ← done in {dt:.2f}s"
            )

        return out_dir

    # ── Save (async with snapshot & CPU PG) ───────────────────────────────────
    @dataclass(slots=True)
    class SaveHandle:
        future: Future
        out_dir: Path
        window: int
        t0: float

        async def wait(self) -> None:
            # Avoid blocking the event loop by running the blocking
            # Future.result() call in a worker thread.
            await asyncio.to_thread(self.future.result)

    async def save_local_async(
        self,
        *,
        model,
        window: int,
        sync_window: int,
        global_step: int,
        topology: str = "FSDP",
    ) -> "DCPCheckpointer.SaveHandle":
        """
        Take a CPU‑offloaded snapshot and kick off an async DCP save.
        Returns a handle you can await before uploading.
        """
        layout = Layout(self.version, window)
        out_dir = self._local_dir(layout)

        # Prepare a stable snapshot (so training can keep going).
        tplr.logger.info(
            f"[DCP][save-async] rank {_rank()}/{_world()} snapshot → (window={window}, dir={_safe(out_dir)})"
        )
        snap = SnapshotState(model)

        # Write the small sidecar immediately.
        if _rank() == 0:
            sidecar = {
                "version": self.version,
                "window": int(window),
                "sync_window": int(sync_window),
                "global_step": int(global_step),
                "world_size_at_save": int(_world()),
                "topology": topology,
                "uid": self.uid,
            }
            (out_dir / "extra_metadata.json").write_text(json.dumps(sidecar, indent=2))

        # Launch async save using a CPU-enabled PG when available.
        # Use TorchTitan/DCP optimizations for better performance
        t0 = time.perf_counter()
        pg = self._cpu_pg if self._cpu_pg is not None else None

        # Create planner with caching for better performance across saves
        planner = DefaultSavePlanner(enable_plan_caching=True)

        fut = async_save(
            state_dict={"app": snap},
            checkpoint_id=str(out_dir),
            process_group=pg,
            async_checkpointer_type=AsyncCheckpointerType.PROCESS,  # Process-based for better performance
            planner=planner,
        )
        tplr.logger.info(
            f"[DCP][save-async] rank {_rank()}/{_world()} launched (window={window})"
        )
        return DCPCheckpointer.SaveHandle(
            future=fut, out_dir=out_dir, window=window, t0=t0
        )

    # ── Upload (per‑rank; rank‑0 handles metadata and pointer) ─────────────────
    async def upload(
        self,
        *,
        window: int,
        background: bool = False,
        delete_local_on_success: bool = True,
        wait_for: "DCPCheckpointer.SaveHandle | None" = None,
        process_group: dist.ProcessGroup | None = None,
        pointer_poll_timeout_s: float = 800.0,
        pointer_poll_interval_s: float = 20.0,
        # New knobs:
        fs_rescan_timeout_s: float = 20.0,
        fs_rescan_interval_s: float = 0.5,
        shared_fs: bool = True,
        mop_up_missing: bool = True,
        pointer_require_all_ranks: bool = True,
        initial_sleep_s: float = 10.0,
    ) -> None:
        layout = Layout(self.version, window)
        local_dir = self._local_dir(layout)

        async def _do() -> None:
            # If the save was launched async, wait for it to complete first.
            if wait_for is not None:
                tplr.logger.info(
                    f"[DCP][upload] rank {_rank(process_group)}/{_world(process_group)} waiting for save (window={window})"
                )
                await wait_for.wait()
                tplr.logger.info(
                    f"[DCP][upload] rank {_rank(process_group)}/{_world(process_group)} save finished (window={window})"
                )

            t_all = time.perf_counter()
            world, r = _world(process_group), _rank(process_group)

            # Small delay to ensure all files are properly written to disk
            # This helps prevent race conditions where files may still be in write buffers
            await asyncio.sleep(initial_sleep_s)

            # Helpers to (re)scan local folder
            def _scan() -> tuple[list[Path], list[Path]]:
                files = [p for p in local_dir.iterdir() if p.is_file()]
                data = [q for q in files if not _is_meta(q)]
                meta = [q for q in files if _is_meta(q)]
                return data, meta

            # Initial scan
            data_files, meta_files = _scan()
            local_owner_ranks = {
                orank
                for p in data_files
                if (orank := _owner_rank_from_name(p.name)) is not None
            }

            # Each rank uploads only files it owns based on the filename encoding
            # DCP filenames include "__{local_pg_rank}_{...}.distcp"
            owned_files = [p for p in data_files if _owner_rank_from_name(p.name) == r]

            # If using a shared FS and we don't see our own shards yet, rescan for a while.
            if shared_fs and not owned_files:
                deadline = time.perf_counter() + fs_rescan_timeout_s
                while time.perf_counter() < deadline and not owned_files:
                    await asyncio.sleep(fs_rescan_interval_s)
                    data_files, meta_files = _scan()
                    owned_files = [
                        p for p in data_files if _owner_rank_from_name(p.name) == r
                    ]
            owned_bytes = sum(p.stat().st_size for p in owned_files)
            meta_bytes = sum(p.stat().st_size for p in meta_files) if r == 0 else 0
            latest_bytes = 0

            tplr.logger.info(
                f"[DCP][upload] rank {r}/{world} start (window={window}) | "
                f"owned shards: {len(owned_files)} (~{_mb(owned_bytes):.2f} MiB)"
                + (
                    f", meta: {len(meta_files)} (~{_mb(meta_bytes):.2f} MiB)"
                    if r == 0
                    else ""
                )
            )

            async def _timed_put(key: str, path: Path, size: int) -> None:
                t0 = time.perf_counter()
                await self.comms.s3_put_object(key=key, file_path=str(path))
                dt = time.perf_counter() - t0
                tplr.logger.info(
                    f"[DCP][upload] rank {r}/{world} ↑ {_safe(key)} "
                    f"{_mb(size):.2f} MiB in {dt:.2f}s ({_mb(size) / dt if dt > 0 else 0:.2f} MiB/s)"
                )

            # Rank‑0 uploads metadata first (pointer comes later).
            tasks: list[asyncio.Task] = []
            if r == 0:
                for p in meta_files:
                    tasks.append(
                        asyncio.create_task(
                            _timed_put(f"{layout.prefix}/{p.name}", p, p.stat().st_size)
                        )
                    )

            # All ranks upload their owned shard files.
            for p in owned_files:
                tasks.append(
                    asyncio.create_task(
                        _timed_put(f"{layout.prefix}/{p.name}", p, p.stat().st_size)
                    )
                )

            if tasks:
                await asyncio.gather(*tasks)

            # Publish the version pointer after we observe shard owners remotely.
            if r == 0:
                # Determine expected owners:
                # - Default: owners actually producing shards (from local folder view)
                # - Optional strict mode: require all ranks 0..world_size_at_save-1
                try:
                    sidecar = json.loads(
                        (local_dir / "extra_metadata.json").read_text()
                    )
                    saved_world = int(sidecar.get("world_size_at_save", world))
                except Exception:
                    saved_world = world
                observed_local_owners = {
                    orank
                    for p in data_files
                    if (orank := _owner_rank_from_name(p.name)) is not None
                }
                if pointer_require_all_ranks:
                    expected_owners = set(range(saved_world))
                else:
                    expected_owners = observed_local_owners or set(range(saved_world))

                bucket = self.comms.bucket
                s3 = await self.comms._get_s3_client(bucket)

                def _owners_from_listing(objs: Iterable[dict]) -> set[int]:
                    owners: set[int] = set()
                    for o in objs or []:
                        key = o.get("Key", "")
                        orank = _owner_rank_from_name(Path(key).name)
                        if orank is not None:
                            owners.add(orank)
                    return owners

                async def _list_remote_all() -> list[dict]:
                    """List all objects under the window prefix (handle pagination)."""
                    contents: list[dict] = []
                    cont: str | None = None
                    while True:
                        args = {"Bucket": bucket.name, "Prefix": f"{layout.prefix}/"}
                        if cont:
                            args["ContinuationToken"] = cont
                        resp = await s3.list_objects_v2(**args)
                        contents.extend(resp.get("Contents", []) or [])
                        if not resp.get("IsTruncated"):
                            break
                        cont = resp.get("NextContinuationToken")
                    return contents

                async def _remote_name_set() -> set[str]:
                    return {
                        Path(o["Key"]).name
                        for o in await _list_remote_all()
                        if o.get("Key")
                    }

                deadline = time.perf_counter() + float(pointer_poll_timeout_s)
                owners_ok = False
                while time.perf_counter() < deadline:
                    contents = await _list_remote_all()
                    owners_remote = _owners_from_listing(contents)
                    if expected_owners.issubset(owners_remote):
                        owners_ok = True
                        break
                    await asyncio.sleep(pointer_poll_interval_s)

                # Mop-up missing shards ONLY after timeout (if needed)
                if not owners_ok and mop_up_missing and shared_fs:
                    # Final mop-up pass after timeout, while others still hold files
                    remote_names = {
                        Path(o["Key"]).name
                        for o in await _list_remote_all()
                        if o.get("Key")
                    }
                    fixups: list[asyncio.Task] = []
                    for p in local_dir.iterdir():
                        if not p.is_file() or _is_meta(p):
                            continue
                        orank = _owner_rank_from_name(p.name)
                        if orank is None or orank not in expected_owners:
                            continue
                        if p.name in remote_names:
                            continue
                        # Upload this missing shard
                        fixups.append(
                            asyncio.create_task(
                                self.comms.s3_put_object(
                                    key=f"{layout.prefix}/{p.name}", file_path=str(p)
                                )
                            )
                        )
                    if fixups:
                        await asyncio.gather(*fixups)

                    # One last check after mop-up
                    contents = await _list_remote_all()
                    owners_remote = _owners_from_listing(contents)
                    owners_ok = expected_owners.issubset(owners_remote)
                    if not owners_ok:
                        tplr.logger.warning(
                            f"[DCP][upload] after mop-up, still missing owners: "
                            f"have={sorted(owners_remote)} expected={sorted(expected_owners)}"
                        )

                tmp = local_dir / "_LATEST.json"
                tmp.write_text(json.dumps({"latest_window": int(window)}, indent=2))
                latest_bytes = tmp.stat().st_size
                await _timed_put(
                    f"checkpoints/{self.version}/_LATEST.json", tmp, latest_bytes
                )
                tplr.logger.info(
                    f"[DCP][upload] rank 0 published pointer for version {self.version} → window {window}"
                )

            # Local cleanup: each rank removes only the files it uploaded.
            if delete_local_on_success:
                # Delete owned data shards for this rank.
                for p in owned_files:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                # Rank‑0 also cleans up metadata and local pointer file.
                if r == 0:
                    for p in meta_files:
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                    try:
                        (local_dir / "_LATEST.json").unlink(missing_ok=True)
                    except Exception:
                        pass
                    # Optionally prune the directory if empty.
                    try:
                        next(local_dir.iterdir())
                    except StopIteration:
                        try:
                            local_dir.rmdir()
                        except Exception:
                            pass

            dt_all = time.perf_counter() - t_all
            total_up_bytes = (
                owned_bytes
                + (meta_bytes if r == 0 else 0)
                + (latest_bytes if r == 0 else 0)
            )
            tplr.logger.info(
                f"[DCP][upload] rank {r}/{world} done in {dt_all:.2f}s | "
                f"bytes≈{total_up_bytes} ({_mb(total_up_bytes):.2f} MiB)"
            )

        if background:
            t = asyncio.create_task(_do())
            self._bg.add(t)
            t.add_done_callback(self._bg.discard)
        else:
            await _do()

    async def flush_background_uploads(self) -> None:
        if self._bg:
            await asyncio.gather(*list(self._bg))
            self._bg.clear()

    # ── Bucket selection (prefer highest-staked validator, fallback to own) ───
    async def _choose_read_bucket(self, prefer_highest_staked: bool = True):
        """
        Returns the bucket to read from. If prefer_highest_staked is True,
        try the highest-staked validator's bucket first; fallback to own bucket.
        """
        if prefer_highest_staked:
            try:
                bucket, _uid = await self.comms._get_highest_stake_validator_bucket()
                if bucket is not None:
                    return bucket
            except Exception:
                pass
        return self.comms.bucket

    # ── Discover latest remote window ──────────────────────────────────────────
    async def _discover_latest(
        self, *, prefer_highest_staked: bool = True
    ) -> int | None:
        """
        Find the newest window, trying highest-staked validator bucket first, then own.
        """
        # Attempt in order: highest-staked → own
        bucket = await self._choose_read_bucket(
            prefer_highest_staked=prefer_highest_staked
        )
        s3 = await self.comms._get_s3_client(bucket)
        # pointer first
        try:
            obj = await self.comms.s3_get_object(
                key=f"checkpoints/{self.version}/_LATEST.json", bucket=bucket
            )
            if isinstance(obj, dict) and "latest_window" in obj:
                return int(obj["latest_window"])
        except Exception:
            pass
        # list fallback
        version_prefix = Layout(self.version).prefix + "/"  # ensure trailing slash
        windows: set[int] = set()
        cont = None
        while True:
            args = {"Bucket": bucket.name, "Prefix": version_prefix}
            if cont:
                args["ContinuationToken"] = cont
            resp = await s3.list_objects_v2(**args)
            for o in resp.get("Contents", []) or []:
                key = o.get("Key", "")
                m = re.match(rf"^{re.escape(version_prefix)}(\d+)/", key)
                if m:
                    windows.add(int(m.group(1)))
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")
        if windows:
            return max(windows)
        return None

    # ── Download: safe default (each rank mirrors full folder) ─────────────────
    async def download_all(
        self, *, window: int | None = None, prefer_highest_staked: bool = True
    ) -> Path | None:
        t_all = time.perf_counter()
        if window is None:
            window = await self._discover_latest(
                prefer_highest_staked=prefer_highest_staked
            )
            if window is None:
                return None
        layout = Layout(self.version, window)
        local_dir = self._local_dir(layout)

        bucket = await self._choose_read_bucket(
            prefer_highest_staked=prefer_highest_staked
        )
        s3 = await self.comms._get_s3_client(bucket)
        tplr.logger.info(
            f"[DCP][download-all] rank {_rank()}/{_world()} start "
            f"(window={window}, bucket={_safe(bucket.name)}, prefix={_safe(layout.prefix)}/)"
        )
        got_any = False
        total_bytes = 0
        cont = None
        while True:
            args = {"Bucket": bucket.name, "Prefix": f"{layout.prefix}/"}
            if cont:
                args["ContinuationToken"] = cont
            resp = await s3.list_objects_v2(**args)
            for obj in resp.get("Contents", []) or []:
                key = obj.get("Key")
                if key:
                    got_any = True
                    size = int(obj.get("Size", 0))
                    t0 = time.perf_counter()
                    await self.comms.s3_get_object(
                        key=key, bucket=bucket, load_data=False, show_progress=False
                    )
                    dt = time.perf_counter() - t0
                    total_bytes += size
                    tplr.logger.debug(
                        f"[DCP][download-all] rank {_rank()}/{_world()} ↓ {_safe(key)} "
                        f"{_mb(size):.2f} MiB in {dt:.2f}s ({_mb(size) / dt if dt > 0 else 0:.2f} MiB/s)"
                    )
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")

        dt_all = time.perf_counter() - t_all
        if got_any:
            tplr.logger.info(
                f"[DCP][download-all] rank {_rank()}/{_world()} done in {dt_all:.2f}s | "
                f"bytes≈{total_bytes} ({_mb(total_bytes):.2f} MiB)"
                f" thru≈{_mb(total_bytes) / dt_all if dt_all > 0 else 0:.2f} MiB/s"
            )

        if got_any:
            return local_dir
        return None

    # ── Download: shared FS optimization (only owners download) ────────────────
    async def download_distributed(
        self,
        *,
        window: int | None = None,
        prefer_highest_staked: bool = True,
        process_group: dist.ProcessGroup | None = None,
    ) -> Path | None:
        """
        Distributed download for a SHARED filesystem:
          • rank‑0 downloads .metadata + JSON sidecars
          • data/shard files assigned by: owner = blake2b(filename) % world_size
          • barrier, then rank‑0 fills any missing files, barrier again
        Result: repo_root/checkpoints/<version>/<window>/... is complete for all ranks.
        """
        t_all = time.perf_counter()
        if window is None:
            window = await self._discover_latest(
                prefer_highest_staked=prefer_highest_staked
            )
            if window is None:
                return None

        layout = Layout(self.version, window)
        local_dir = self._local_dir(layout)

        world, r = _world(), _rank()
        # Try highest-staked bucket first, then own
        bucket = await self._choose_read_bucket(
            prefer_highest_staked=prefer_highest_staked
        )
        s3 = await self.comms._get_s3_client(bucket)
        # 1) Enumerate keys under the window prefix
        keys: list[tuple[str, int]] = []
        cont = None
        while True:
            args = {"Bucket": bucket.name, "Prefix": f"{layout.prefix}/"}
            if cont:
                args["ContinuationToken"] = cont
            resp = await s3.list_objects_v2(**args)
            for o in resp.get("Contents", []) or []:
                if k := o.get("Key"):
                    keys.append((k, int(o.get("Size", 0))))
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")

        # 2) Partition work (round‑robin by filename for data shards; rank‑0 handles metadata)
        meta_keys: list[tuple[str, int]] = [
            (k, sz) for (k, sz) in keys if _is_meta(Path(k).name)
        ]
        data_keys: list[tuple[str, int]] = [
            (k, sz) for (k, sz) in keys if not _is_meta(Path(k).name)
        ]
        data_keys_sorted = sorted(data_keys, key=lambda pair: Path(pair[0]).name)
        assigned: list[tuple[str, int]] = [
            pair for i, pair in enumerate(data_keys_sorted) if i % world == r
        ]
        if r == 0:
            assigned.extend(meta_keys)  # rank‑0 also pulls metadata/JSON

        owned_bytes = sum(sz for _, sz in assigned if not _is_meta(Path(_).name))
        meta_bytes = sum(sz for _, sz in meta_keys) if r == 0 else 0
        tplr.logger.info(
            f"[DCP][download-dist] rank {r}/{world} start (window={window}, bucket={bucket.name}) | "
            f"assigned: {len(assigned)} (~{_mb(owned_bytes):.2f} MiB)"
            + (f", meta-by-r0: ~{_mb(meta_bytes):.2f} MiB" if r == 0 else "")
        )

        # 3) Owners download their files (skip if present)
        for key, sz in assigned:
            dst = self.repo_root / key  # mirrored path
            if dst.exists():
                tplr.logger.debug(
                    f"[DCP][download-dist] rank {r}/{world} skip (exists) {_safe(key)}"
                )
                continue
            t0 = time.perf_counter()
            await self.comms.s3_get_object(
                key=key, bucket=bucket, load_data=False, show_progress=True
            )
            dt = time.perf_counter() - t0
            tplr.logger.debug(
                f"[DCP][download-dist] rank {r}/{world} ↓ {_safe(key)} "
                f"{_mb(sz):.2f} MiB in {dt:.2f}s ({_mb(sz) / dt if dt > 0 else 0:.2f} MiB/s)"
            )

        _barrier(process_group)  # all owners finished

        # 4) Rank‑0 mop‑up (if any file is still missing, download it)
        if r == 0:
            mop_files = 0
            mop_bytes = 0
            for key, sz in keys:
                dst = self.repo_root / key
                if not dst.exists():
                    t0 = time.perf_counter()
                    await self.comms.s3_get_object(
                        key=key, bucket=bucket, load_data=False, show_progress=True
                    )
                    dt = time.perf_counter() - t0
                    mop_files += 1
                    mop_bytes += sz
                    tplr.logger.debug(
                        f"[DCP][download-dist] rank 0 mop-up ↓ {_safe(key)} "
                        f"{_mb(sz):.2f} MiB in {dt:.2f}s ({_mb(sz) / dt if dt > 0 else 0:.2f} MiB/s)"
                    )
            tplr.logger.info(
                f"[DCP][download-dist] rank 0 mop-up completed | files={mop_files}, "
                f"bytes≈{mop_bytes} ({_mb(mop_bytes):.2f} MiB)"
            )

        _barrier(process_group)  # ensure folder is complete for all ranks

        dt_all = time.perf_counter() - t_all
        total_assigned = sum(sz for _, sz in assigned) + (meta_bytes if r == 0 else 0)
        tplr.logger.info(
            f"[DCP][download-dist] rank {r}/{world} done in {dt_all:.2f}s | "
            f"bytes≈{total_assigned} ({_mb(total_assigned):.2f} MiB) "
            f"thru≈{_mb(total_assigned) / dt_all if dt_all > 0 else 0:.2f} MiB/s"
        )
        return local_dir

    # ── Load (reshard automatically to current topology) ───────────────────────
    def load_local(
        self, *, model, window: int, process_group: dist.ProcessGroup | None = None
    ) -> None:
        layout = Layout(self.version, window)
        ckpt_dir = self._local_dir(layout)
        state = {"app": AppState(model)}
        load(state_dict=state, checkpoint_id=str(ckpt_dir), process_group=process_group)

    async def download_and_load(
        self,
        *,
        model,
        window: int | None = None,
        shared_fs: bool = True,
        process_group: dist.ProcessGroup | None = None,
        prefer_highest_staked: bool = True,
    ) -> tuple[int, int] | None:
        local_dir = await (
            self.download_distributed(
                window=window, prefer_highest_staked=prefer_highest_staked
            )
            if shared_fs
            else self.download_all(
                window=window, prefer_highest_staked=prefer_highest_staked
            )
        )
        if local_dir is None:
            return None
        sidecar = json.loads((local_dir / "extra_metadata.json").read_text())
        w = int(sidecar["window"])
        global_step = int(sidecar.get("global_step", -1))
        self.load_local(model=model, window=w, process_group=process_group)
        return w, global_step

    async def check_checkpoint_exists(
        self, *, window: int, prefer_highest_staked: bool = True
    ) -> bool:
        """Check if a checkpoint exists without downloading it.

        Args:
            window: Window number to check
            prefer_highest_staked: Whether to check highest-staked validator's bucket first

        Returns:
            True if checkpoint exists and appears complete, False otherwise
        """
        try:
            bucket = await self._choose_read_bucket(
                prefer_highest_staked=prefer_highest_staked
            )
            s3 = await self.comms._get_s3_client(bucket)

            # Check if the window directory exists with essential files
            prefix = f"checkpoints/{self.version}/{window}/"

            resp = await s3.list_objects_v2(
                Bucket=bucket.name,
                Prefix=prefix,
                MaxKeys=1000,
            )

            objects = resp.get("Contents", [])
            if not objects:
                return False

            # Check for essential files
            has_metadata = any(".metadata" in obj.get("Key", "") for obj in objects)
            has_sidecar = any(
                "extra_metadata.json" in obj.get("Key", "") for obj in objects
            )

            # Determine expected world size from sidecar if possible
            expected_world = None
            try:
                sidecar_obj = await self.comms.s3_get_object(
                    key=f"{prefix}extra_metadata.json", bucket=bucket
                )
                if isinstance(sidecar_obj, (bytes, str)):
                    meta = json.loads(
                        sidecar_obj
                        if isinstance(sidecar_obj, str)
                        else sidecar_obj.decode("utf-8")
                    )
                else:
                    meta = sidecar_obj
                expected_world = int(meta.get("world_size_at_save"))
            except Exception:
                pass

            owner_ranks = set()
            for o in objects:
                key = o.get("Key", "")
                orank = _owner_rank_from_name(Path(key).name)
                if orank is not None:
                    owner_ranks.add(orank)

            all_ranks_present = expected_world is None or all(
                rk in owner_ranks for rk in range(expected_world)
            )

            if not all_ranks_present:
                tplr.logger.warning(
                    f"Checkpoint at window {window} may be incomplete. "
                    f"Owners observed: {sorted(owner_ranks)}; expected_world={expected_world}"
                )

            return has_metadata and has_sidecar and all_ranks_present

        except Exception as e:
            tplr.logger.warning(f"Error checking checkpoint existence: {_safe(e)}")
            return False

    def cleanup_local_checkpoints(self, keep_latest: int = 1) -> None:
        """
        Remove old local checkpoint directories, keeping only the latest N windows.

        Args:
            keep_latest: Number of latest checkpoints to keep (default: 1)
        """
        checkpoint_base = self.repo_root / "checkpoints" / self.version

        if not checkpoint_base.exists():
            return

        # Get all window directories and extract window numbers
        window_dirs = []
        for d in checkpoint_base.iterdir():
            if d.is_dir() and d.name.isdigit():
                try:
                    window_num = int(d.name)
                    window_dirs.append((window_num, d))
                except ValueError:
                    continue

        # Sort by window number (ascending)
        window_dirs.sort(key=lambda x: x[0])

        # Remove all but the latest keep_latest checkpoints
        if len(window_dirs) > keep_latest:
            for _, old_checkpoint in window_dirs[:-keep_latest]:
                try:
                    shutil.rmtree(old_checkpoint)
                    if _rank() == 0:
                        tplr.logger.info(
                            f"[DCP] Removed old checkpoint: {_safe(old_checkpoint)}"
                        )
                except Exception as e:
                    if _rank() == 0:
                        tplr.logger.warning(
                            f"[DCP] Failed to remove {_safe(old_checkpoint)}: {_safe(e)}"
                        )
        else:
            if _rank() == 0:
                tplr.logger.info(
                    f"[DCP] Checkpoint cleanup: {len(window_dirs)} checkpoints present, "
                    f"keeping all (threshold: {keep_latest})"
                )
