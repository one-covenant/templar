import abc
import asyncio
import functools
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import bittensor as bt
import torch.profiler as tp
import websockets.exceptions  # ensure import before threads start
from bittensor.core.subtensor import ScaleObj

import tplr
from tplr.distributed import dist_helper

CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))


class BaseNode(abc.ABC):
    # ――― attributes that subclasses will overwrite ―――
    executor: ThreadPoolExecutor
    world_size: int = 1
    comms: tplr.Comms
    config: Any = None
    hparams: SimpleNamespace
    subtensor: bt.Subtensor
    ckpt: tplr.DCPCheckpointer

    # background bookkeeping
    stop_event: asyncio.Event

    # ─── window-change helpers ────────────────────────────────────────────
    window_changed: asyncio.Event | None
    _notify_loop: asyncio.AbstractEventLoop | None

    def __init__(self):
        # -------- shared state ------------------------------------------------
        self.stop_event = asyncio.Event()
        self._bg_tasks: set[asyncio.Task] = set()
        self._threads: list[threading.Thread] = []

        # window signalling (initialised in main)
        self.window_changed = None
        self._notify_loop = None

        # ---- things your subclasses already set -----------------------------
        self.current_block = 0
        self.current_window = 0
        self.subtensor_rpc: bt.Subtensor | None = None
        self.subtensor_client: bt.Subtensor | None = None

        # ── optional torch profiler (initialised in main) ────────────────
        self._prof = None
        # self.config, self.world_size …  come from the concrete node

        # Continue MRO chain to initialize parent classes (e.g., Trainer)
        super().__init__()

    async def main(self):
        loop = asyncio.get_running_loop()
        self._setup_signal_handlers(loop)

        # event-driven notifications whenever current_window increments
        self.window_changed = asyncio.Event()
        self._notify_loop = loop

        # start the block-listener *thread*
        t = threading.Thread(target=self.block_listener, name="blocks", daemon=True)
        t.start()
        self._threads.append(t)

        # ── profiler setup (enabled only when flag > 0) ─────────────────
        prof_iters = int(getattr(self.config, "profile_iters", 0))
        if prof_iters > 0:
            # Profile all ranks, with separate directories for each
            profile_dir = getattr(self.config, "profile_dir", "./log/profiler")
            rank = getattr(self, "rank", 0)
            profile_dir = f"{profile_dir}/rank_{rank}"
            os.makedirs(profile_dir, exist_ok=True)

            # Store profiler config but don't start it yet
            self._prof_config = {
                "activities": [tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA],
                "schedule": tp.schedule(wait=0, warmup=1, active=prof_iters, repeat=1),
                "on_trace_ready": tp.tensorboard_trace_handler(profile_dir),
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": True,
            }
            self._prof = None  # Will be created in inner_steps
        else:
            self._prof_config = None
            self._prof = None

        # subclasses do their normal work here ----------------------------
        try:
            await self.run()
        except Exception:
            # Catch and log the unhandled exception before shutting down.
            tplr.logger.error("Unhandled exception in run()", exc_info=True)
        finally:
            # ensure we exit cleanly even if run() returns without CTRL-C
            if not self.stop_event.is_set():
                await self._graceful_shutdown(signal.SIGTERM)

    @abc.abstractmethod
    async def run(self):
        raise NotImplementedError

    # ──────────────────────────────────────────────────────────────────────
    #  Shared helper: wait until the chain reaches a specific window
    # ──────────────────────────────────────────────────────────────────────
    async def wait_until_window(self, target_window: int) -> None:
        """
        Block until ``self.current_window`` ≥ *target_window*.
        Usable by both validators and miners.
        """
        evt = self.window_changed

        # ── clear a leftover signal ──────────────────────────────────────
        # If the previous window increment already set the flag and nobody
        # has waited on it yet, we’d wake up immediately and log twice.
        if evt is not None and evt.is_set():
            evt.clear()

        while not self.stop_event.is_set():
            if self.current_window >= target_window:
                return

            # ── log an approximate ETA ──────────────────────────────────
            #   • how many whole windows still to go
            remaining_windows = target_window - self.current_window

            #   • how many blocks already elapsed in the *current* window
            blocks_into_window = self.current_block % self.hparams.blocks_per_window

            #   • total blocks remaining until the target window begins
            remaining_blocks = (
                remaining_windows * self.hparams.blocks_per_window - blocks_into_window
            )

            #   • assuming ≈12 s per block
            eta_seconds = max(0, remaining_blocks * 12)
            mins, secs = divmod(int(eta_seconds), 60)

            tplr.logger.info(
                f"⏳ waiting for window {target_window} "
                f"(~{mins} m {secs:02d} s, {remaining_blocks} blocks)"
            )

            if evt is None:  # should not happen
                await asyncio.sleep(0.5)
            else:
                await evt.wait()
                evt.clear()

    def _setup_signal_handlers(self, loop: asyncio.AbstractEventLoop):
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                functools.partial(asyncio.create_task, self._graceful_shutdown(sig)),
            )

    async def _graceful_shutdown(self, sig):
        tplr.logger.warning(f"↩  received {sig.name} – shutting down …")
        self.stop_event.set()
        await self.ckpt.flush_background_uploads()

        # cancel background asyncio tasks
        for task in list(self._bg_tasks):
            task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)

        # wait for helper threads
        for th in self._threads:
            th.join(timeout=2)

        # close shared subtensor client
        if self.subtensor_rpc is not None:
            try:
                self.subtensor_rpc.close()
            except Exception:
                pass

        if self.subtensor_client is not None:
            try:
                self.subtensor_client.close()
            except Exception:
                pass

        # external resources
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, "world_size"):
            dist_helper.destroy_process_group()

        await asyncio.sleep(0.2)  # allow log flush
        tplr.logger.info("✔ shutdown complete")

    def query_block_timestamp(
        self,
        block: int,
        *,
        retries: int = 5,
        init_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> float | None:
        """
        Return the Unix-epoch timestamp (seconds) for `block` or None.

        • Uses a *new* Subtensor rpc client each try, so there is
          no clash with the streaming websocket in `block_listener`.
        • Retries with exponential back-off.
        """
        if self.subtensor_client is None:
            st = bt.subtensor(config=self.config)
            self.subtensor_client = st

        delay = init_delay
        for attempt in range(1, retries + 1):
            try:
                resp = self.subtensor_client.query_module(
                    "Timestamp", "Now", block=block
                )
                if resp is None or not isinstance(resp, ScaleObj):
                    raise ValueError(f"Could not query timestamp for {block}")
                ts_value = (
                    cast(int, resp.value) / 1000
                )  # convert milliseconds to seconds
                return ts_value
            except Exception as e:
                tplr.logger.warning(
                    f"[timestamp] block {block} attempt {attempt}/{retries} failed: {e}"
                )
                if attempt == retries:
                    return None

                # reconnect once before the next retry
                try:
                    self.subtensor_client.substrate.close()
                except Exception:
                    pass
                self.subtensor_client.substrate.initialize()
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

    def block_listener(self):
        backoff, max_backoff = 1, 60

        # create it once
        if self.subtensor_rpc is None:
            self.subtensor_rpc = bt.subtensor(config=self.config)

        def handler(event):
            try:
                self.current_block = int(event["header"]["number"])
                new_window = self.current_block // self.hparams.blocks_per_window
                if new_window != self.current_window:
                    self.current_window = new_window
                    if hasattr(self, "comms"):
                        self.comms.current_window = self.current_window
                    tplr.logger.info(f"▶ window → {self.current_window}")

                    # notify any awaiters
                    if self.window_changed and self._notify_loop:
                        self._notify_loop.call_soon_threadsafe(self.window_changed.set)
            except Exception as e:
                tplr.logger.error(f"block-handler err: {e}")

        while not self.stop_event.is_set():
            try:
                # re‑open the socket instead of instantiating a new client
                self.subtensor_rpc.substrate.initialize()
                self.subtensor_rpc.substrate.subscribe_block_headers(handler)
                backoff = 1  # reset back‑off on success
            except websockets.exceptions.ConnectionClosedError as e:
                if self.stop_event.is_set():
                    break
                tplr.logger.warning(f"ws closed: {e} – retrying in {backoff}s")
            except Exception as e:
                if self.stop_event.is_set():
                    break
                tplr.logger.error(
                    f"block subscription err: {e} – retrying in {backoff}s"
                )
            finally:
                # make sure the helper thread inside Substrate is stopped cleanly
                try:
                    self.subtensor_rpc.substrate.close()
                except Exception:
                    pass

            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
