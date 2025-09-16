"""Integration test for the Evaluator.

1. Properly detecting new checkpoints by window number
2. Skipping previously evaluated checkpoints
3. Discovering new checkpoints
4. Updating tracking state correctly
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurons.evaluator import Evaluator


def setup_evaluator_with_mocks():
    """Setup evaluator with necessary mocks for testing"""
    with (
        patch("tplr.distributed.dist_helper.init_process_group"),
        patch("tplr.distributed.dist_helper.is_master", return_value=True),
        patch("tplr.distributed.dist_helper.rank", 0),
        patch("tplr.distributed.dist_helper.world_size", 2),
        patch("tplr.distributed.dist_helper.local_rank", 0),
        patch("tplr.distributed.dist_helper.device", "cpu"),
        patch("tplr.load_hparams"),
        patch("tplr.comms.Comms"),
        patch("tplr.DCPCheckpointer"),
        patch("tplr.initialize_wandb"),
        patch("tplr.metrics.MetricsLogger"),
        patch("neurons.evaluator.Evaluator._initialize_or_load_model"),
        patch("torch.distributed.broadcast"),
    ):
        evaluator = Evaluator.__new__(Evaluator)

        # Setup basic attributes
        evaluator.config = MagicMock(
            netuid=3,
            device="cpu",
            eval_interval=30,
            version="test_version",
            cache_dir="/tmp/cache",
            tasks="arc_easy",
            from_window=None,
            to_window=None,
            no_follow=False,
            force_reval=False,
            custom_eval_path=None,  # Add this to prevent TypeError
        )
        evaluator.netuid = 3
        evaluator.version = "test_version"
        evaluator.eval_interval = 30

        # Setup backfill & control flags
        evaluator.from_window = None
        evaluator.to_window = None
        evaluator.no_follow = False
        evaluator.force_reval = False

        # Setup distributed attributes
        evaluator.rank = 0
        evaluator.world_size = 2
        evaluator.local_rank = 0
        evaluator.is_master = True
        evaluator.device = "cpu"

        # Setup model and components
        evaluator.model = MagicMock()
        evaluator.metrics_logger = MagicMock()
        evaluator.comms = MagicMock()
        evaluator.hparams = MagicMock()
        evaluator.hparams.blocks_per_window = 100
        evaluator.hparams.checkpoint_init_version = (
            None  # No bootstrap version by default
        )
        evaluator.ckpt = MagicMock()
        evaluator.model_cache = MagicMock()

        # Setup state tracking
        evaluator.evaluated_windows = set()
        evaluator.baseline_evaluated = False
        evaluator.last_discovered_window = None
        evaluator.start_window = 0

        # Setup wandb mock
        evaluator.wandb = MagicMock()

        # Note: Don't mock _resolve_to_window and backfill_range here
        # They need to be tested as real methods in some tests

        yield evaluator


@pytest.fixture
def evaluator():
    """
    Fixture to return a properly mocked evaluator instance
    """
    yield from setup_evaluator_with_mocks()


@pytest.mark.asyncio
async def test_evaluator_skips_old_checkpoints(evaluator):
    """
    Test that check_latest_checkpoint skips already evaluated checkpoints
    """
    # Mark window 100 as already evaluated
    evaluator.evaluated_windows.add(100)

    # Mock _discover_latest to return window 100 (already evaluated)
    evaluator.ckpt._discover_latest = AsyncMock(return_value=100)

    # Mock check_checkpoint_exists to return True (complete)
    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=True)

    # Call check_latest_checkpoint
    result = await evaluator.check_latest_checkpoint()

    # Verify that _discover_latest was called with correct parameters
    evaluator.ckpt._discover_latest.assert_called_once_with(prefer_highest_staked=True)

    # Should return None since window 100 is already evaluated
    assert result is None, "Should return None for already evaluated windows"


@pytest.mark.asyncio
async def test_evaluator_loads_new_checkpoints(evaluator):
    """
    Test that check_latest_checkpoint discovers new checkpoints not yet evaluated
    """
    # No windows evaluated yet
    evaluator.evaluated_windows = set()

    # Mock _discover_latest to return a new checkpoint at window 110
    evaluator.ckpt._discover_latest = AsyncMock(return_value=110)

    # Mock check_checkpoint_exists to return True (complete)
    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=True)

    # Call check_latest_checkpoint
    result = await evaluator.check_latest_checkpoint()

    # Verify that _discover_latest was called with correct parameters
    evaluator.ckpt._discover_latest.assert_called_once_with(prefer_highest_staked=True)

    assert result == 110, "Should return the new window number"
    assert 110 not in evaluator.evaluated_windows, "Should not mark as evaluated yet"


@pytest.mark.asyncio
async def test_evaluate_window_successful(evaluator):
    """
    Test successful evaluation of a checkpoint window
    """
    # Setup mocks
    evaluator.ckpt.download_and_load = AsyncMock(return_value=(120, 20))
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_120")
    evaluator.run_custom_eval = MagicMock(return_value=(5.2, 1.8))
    evaluator.run_lm_eval_multi_gpu = MagicMock(return_value={"arc_easy": 0.75})
    evaluator.model_cache = MagicMock()
    evaluator.tasks = ["arc_easy"]

    # Mock pause_ddp_for_lm_eval context manager
    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        success = await evaluator.evaluate_window(120, is_baseline=False)

    assert success, "Should return True for successful evaluation"
    assert 120 in evaluator.evaluated_windows, "Should mark window as evaluated"
    evaluator.ckpt.download_and_load.assert_called_once()
    evaluator.save_model_for_eval.assert_called_once_with(120)
    evaluator.run_custom_eval.assert_called_once_with(120, 20)


@pytest.mark.asyncio
async def test_evaluate_window_skips_already_evaluated(evaluator):
    """
    Test that evaluate_window skips already evaluated windows
    """
    # Mark window 100 as already evaluated
    evaluator.evaluated_windows.add(100)

    success = await evaluator.evaluate_window(100)

    assert success, "Should return True for already evaluated window"
    # Should not call any evaluation methods
    evaluator.ckpt.download_and_load.assert_not_called()


@pytest.mark.asyncio
async def test_evaluate_window_baseline(evaluator):
    """
    Test baseline evaluation (no checkpoint loading)
    """
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_baseline")
    evaluator.run_custom_eval = MagicMock(return_value=(10.5, 2.3))
    evaluator.run_lm_eval_multi_gpu = MagicMock(return_value={"arc_easy": 0.25})
    evaluator.model_cache = MagicMock()
    evaluator.tasks = ["arc_easy"]
    evaluator.start_window = 0

    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        success = await evaluator.evaluate_window(0, is_baseline=True)

    assert success, "Should return True for successful baseline evaluation"
    assert 0 in evaluator.evaluated_windows, "Should mark window 0 as evaluated"
    # Should not try to download checkpoint for baseline
    evaluator.ckpt.download_and_load.assert_not_called()
    evaluator.save_model_for_eval.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_evaluate_window_download_failure(evaluator):
    """
    Test handling of checkpoint download failure
    """
    evaluator.ckpt.download_and_load = AsyncMock(return_value=None)

    success = await evaluator.evaluate_window(130, is_baseline=False)

    assert not success, "Should return False when checkpoint download fails"
    assert 130 not in evaluator.evaluated_windows, (
        "Should not mark as evaluated on failure"
    )


@pytest.mark.asyncio
async def test_model_cache_exists(evaluator):
    """
    Test model cache functionality when model already cached
    """
    from pathlib import Path

    from neurons.evaluator import ModelCache

    cache = ModelCache(Path("/tmp/test_cache"))
    evaluator.model_cache = cache

    # Mock cache.exists to return True
    with patch.object(cache, "exists", return_value=True):
        with patch.object(cache, "get_path", return_value=Path("/tmp/cached_model")):
            result = evaluator.save_model_for_eval(100)

    assert result == Path("/tmp/cached_model"), "Should return cached path"


@pytest.mark.asyncio
async def test_run_initial_baseline(evaluator):
    """
    Test run() method with no checkpoints (baseline evaluation)
    """
    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.check_latest_checkpoint = AsyncMock(return_value=None)
    evaluator.evaluate_window = AsyncMock(return_value=True)
    evaluator.baseline_evaluated = False

    # Mock asyncio.sleep to prevent infinite loop
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        # Set side effect to raise exception after first iteration
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        try:
            await evaluator.run()
        except KeyboardInterrupt:
            pass

    # Should evaluate baseline
    evaluator.evaluate_window.assert_called_with(0, is_baseline=True)
    assert evaluator.start_window == 0


@pytest.mark.asyncio
async def test_run_with_checkpoint(evaluator):
    """
    Test run() method with existing checkpoint
    """
    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.check_latest_checkpoint = AsyncMock(return_value=150)
    evaluator.evaluate_window = AsyncMock(return_value=True)
    evaluator.baseline_evaluated = False

    # Mock asyncio.sleep to prevent infinite loop
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        try:
            await evaluator.run()
        except KeyboardInterrupt:
            pass

    # Should evaluate checkpoint 150
    evaluator.evaluate_window.assert_called_with(150)


@pytest.mark.asyncio
async def test_run_loop_discovers_new_checkpoint(evaluator):
    """
    Test main loop discovering and evaluating new checkpoint
    """
    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.evaluate_window = AsyncMock(return_value=True)
    evaluator.config.eval_interval = 1  # Short interval for testing

    # First check returns None, second check returns 160
    evaluator.check_latest_checkpoint = AsyncMock(
        side_effect=[None, 160, KeyboardInterrupt()]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with patch("tplr.distributed.dist_helper.safe_barrier"):
            try:
                await evaluator.run()
            except KeyboardInterrupt:
                pass

    # Should have evaluated window 160
    assert evaluator.evaluate_window.call_count >= 1
    calls = evaluator.evaluate_window.call_args_list
    assert any(call[0][0] == 160 for call in calls), "Should evaluate window 160"


@pytest.mark.asyncio
async def test_check_latest_checkpoint_with_incomplete_checkpoint(evaluator):
    """
    Test that check_latest_checkpoint waits for complete checkpoints
    """
    # Mock _discover_latest to return window 200
    evaluator.ckpt._discover_latest = AsyncMock(return_value=200)

    # Mock check_checkpoint_exists to return False (incomplete)
    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=False)

    # Call check_latest_checkpoint
    result = await evaluator.check_latest_checkpoint()

    # Should return None since checkpoint is incomplete
    assert result is None, "Should return None for incomplete checkpoint"

    # Verify both methods were called
    evaluator.ckpt._discover_latest.assert_called_once_with(prefer_highest_staked=True)
    evaluator.ckpt.check_checkpoint_exists.assert_called_once_with(window=200)


@pytest.mark.asyncio
async def test_check_latest_checkpoint_with_complete_checkpoint(evaluator):
    """
    Test that check_latest_checkpoint returns window when checkpoint is complete
    """
    # Mock _discover_latest to return window 210
    evaluator.ckpt._discover_latest = AsyncMock(return_value=210)

    # Mock check_checkpoint_exists to return True (complete)
    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=True)

    # Call check_latest_checkpoint
    result = await evaluator.check_latest_checkpoint()

    # Should return the window number
    assert result == 210, "Should return window number for complete checkpoint"

    # Verify both methods were called
    evaluator.ckpt._discover_latest.assert_called_once_with(prefer_highest_staked=True)
    evaluator.ckpt.check_checkpoint_exists.assert_called_once_with(window=210)


@pytest.mark.asyncio
async def test_run_with_bootstrap_version_no_checkpoint(evaluator):
    """
    Test run() method with bootstrap version configured but no checkpoints
    """
    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.check_latest_checkpoint = AsyncMock(return_value=None)
    evaluator.evaluate_window = AsyncMock(return_value=True)
    evaluator.baseline_evaluated = False

    # Set bootstrap version to skip baseline
    evaluator.hparams.checkpoint_init_version = "v2.0.0"

    # Mock asyncio.sleep to prevent infinite loop
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        try:
            await evaluator.run()
        except KeyboardInterrupt:
            pass

    # Should NOT evaluate baseline when bootstrap is configured
    evaluator.evaluate_window.assert_not_called()
    assert evaluator.start_window == 0


def test_model_cache_cleanup():
    """
    Test ModelCache cleanup method removes old models
    """
    import tempfile
    from pathlib import Path

    from neurons.evaluator import ModelCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ModelCache(Path(tmpdir))

        # Create mock model directories
        for window in [100, 200, 300, 400]:
            model_dir = cache.get_path(window)
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").touch()
            (model_dir / "model.safetensors").touch()

        # Keep only latest 2
        cache.cleanup(keep_latest=2)

        # Check that only windows 300 and 400 remain
        remaining = sorted(
            [
                int(d.name.split("_")[1])
                for d in cache.base_dir.iterdir()
                if d.is_dir() and d.name.startswith("window_")
            ]
        )
        assert remaining == [300, 400], f"Expected [300, 400] but got {remaining}"


def test_model_cache_exists():
    """
    Test ModelCache exists method correctly identifies cached models
    """
    import tempfile
    from pathlib import Path

    from neurons.evaluator import ModelCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ModelCache(Path(tmpdir))

        # Test non-existent model
        assert not cache.exists(100), "Should return False for non-existent model"

        # Create model directory without files
        model_dir = cache.get_path(200)
        model_dir.mkdir(parents=True)
        assert not cache.exists(200), "Should return False without model files"

        # Add config but no weights
        (model_dir / "config.json").touch()
        assert not cache.exists(200), "Should return False without weights"

        # Add weights
        (model_dir / "model.safetensors").touch()
        assert cache.exists(200), "Should return True with config and weights"

        # Test with pytorch_model.bin instead
        model_dir2 = cache.get_path(300)
        model_dir2.mkdir(parents=True)
        (model_dir2 / "config.json").touch()
        (model_dir2 / "pytorch_model.bin").touch()
        assert cache.exists(300), "Should return True with pytorch_model.bin"


def test_loss_to_bpb():
    """
    Test _loss_to_bpb conversion function
    """
    import math

    from neurons.evaluator import _loss_to_bpb

    # Test with total_bytes
    bpb = _loss_to_bpb(100.0, 50, total_bytes=100)
    expected = (100.0 / 50) / math.log(2) * (50 / 100)
    assert abs(bpb - expected) < 0.001, f"Expected {expected}, got {bpb}"

    # Test with bytes_per_token
    bpb = _loss_to_bpb(100.0, 50, bytes_per_token=2.0)
    expected = (100.0 / 50) / math.log(2) * (1.0 / 2.0)
    assert abs(bpb - expected) < 0.001, f"Expected {expected}, got {bpb}"

    # Test with zero tokens
    bpb = _loss_to_bpb(100.0, 0, total_bytes=100)
    assert bpb == float("inf"), "Should return inf for zero tokens"

    # Test missing params raises error
    with pytest.raises(ValueError):
        _loss_to_bpb(100.0, 50)


@pytest.mark.asyncio
async def test_evaluate_window_cleanup_called(evaluator):
    """
    Test that cleanup methods are called after evaluation
    """
    evaluator.evaluated_windows = {100, 200}  # Already have 2 evaluations
    evaluator.ckpt.download_and_load = AsyncMock(return_value=(300, 30))
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_300")
    evaluator.run_custom_eval = MagicMock(return_value=(5.2, 1.8))
    evaluator.run_lm_eval_multi_gpu = MagicMock(return_value={"arc_easy": 0.75})
    evaluator.model_cache = MagicMock()
    evaluator.ckpt.cleanup_local_checkpoints = MagicMock()
    evaluator.tasks = ["arc_easy"]

    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        success = await evaluator.evaluate_window(300, is_baseline=False)

    assert success
    assert 300 in evaluator.evaluated_windows

    # Should call cleanup methods since we have evaluations
    evaluator.model_cache.cleanup.assert_called_once_with(keep_latest=1)
    evaluator.ckpt.cleanup_local_checkpoints.assert_called_once_with(keep_latest=1)


@pytest.mark.asyncio
async def test_evaluate_window_global_step_calculation(evaluator):
    """
    Test global_step calculation when not in checkpoint sidecar
    """
    evaluator.start_window = 100
    evaluator.ckpt.download_and_load = AsyncMock(
        return_value=(150, -1)
    )  # -1 means no global_step
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_150")
    evaluator.run_custom_eval = MagicMock(return_value=(5.2, 1.8))
    evaluator.run_lm_eval_multi_gpu = MagicMock(return_value={})
    evaluator.model_cache = MagicMock()
    evaluator.tasks = []

    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        success = await evaluator.evaluate_window(150, is_baseline=False)

    assert success
    # run_custom_eval should be called with calculated global_step
    evaluator.run_custom_eval.assert_called_once_with(150, 50)  # 150 - 100 = 50


@pytest.mark.asyncio
async def test_evaluate_window_mmlu_every_fourth(evaluator):
    """
    Test that MMLU with few-shot is run every 4th evaluation
    """
    # Set up so that MMLU will run (len(evaluated_windows) % 4 == 0)
    # We need 0, 4, 8, 12, etc. windows already evaluated
    # Let's use 0 windows (empty set) so the first evaluation triggers MMLU
    evaluator.evaluated_windows = set()  # Empty, so len() % 4 == 0
    evaluator.tasks = ["arc_easy", "mmlu"]
    evaluator.ckpt.download_and_load = AsyncMock(return_value=(130, 30))
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_130")
    evaluator.run_custom_eval = MagicMock(return_value=(5.2, 1.8))
    evaluator.run_lm_eval_multi_gpu = MagicMock(
        return_value={"arc_easy": 0.75, "mmlu": 0.60}
    )
    evaluator.model_cache = MagicMock()

    # Mock the sentinel file context
    mock_sentinel = MagicMock()
    mock_sentinel.touch = MagicMock()

    with patch("neurons.evaluator.pause_ddp_for_lm_eval") as mock_pause:
        mock_pause.return_value.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_pause.return_value.__exit__ = MagicMock(return_value=None)

        success = await evaluator.evaluate_window(130, is_baseline=False)

    assert success
    # Should be called twice: once for regular tasks, once for MMLU with few-shot
    assert evaluator.run_lm_eval_multi_gpu.call_count == 2

    # Check the calls
    calls = evaluator.run_lm_eval_multi_gpu.call_args_list
    # First call should be regular tasks without mmlu
    assert "arc_easy" in calls[0][1]["tasks"]
    assert "mmlu" not in calls[0][1]["tasks"]

    # Second call should be MMLU with few-shot
    assert calls[1][1]["tasks"] == ["mmlu"]
    assert calls[1][1]["num_fewshot"] == 5


@pytest.mark.asyncio
async def test_evaluate_window_mmlu_not_fourth(evaluator):
    """
    Test that MMLU with few-shot is NOT run when not the 4th evaluation
    """
    # Set up so that MMLU will NOT run (len(evaluated_windows) % 4 != 0)
    evaluator.evaluated_windows = {100, 110}  # 2 windows, so len() % 4 == 2
    evaluator.tasks = ["arc_easy", "mmlu"]
    evaluator.ckpt.download_and_load = AsyncMock(return_value=(130, 30))
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model_130")
    evaluator.run_custom_eval = MagicMock(return_value=(5.2, 1.8))
    evaluator.run_lm_eval_multi_gpu = MagicMock(return_value={"arc_easy": 0.75})
    evaluator.model_cache = MagicMock()

    # Mock the sentinel file context
    mock_sentinel = MagicMock()
    mock_sentinel.touch = MagicMock()

    with patch("neurons.evaluator.pause_ddp_for_lm_eval") as mock_pause:
        mock_pause.return_value.__enter__ = MagicMock(return_value=mock_sentinel)
        mock_pause.return_value.__exit__ = MagicMock(return_value=None)

        success = await evaluator.evaluate_window(130, is_baseline=False)

    assert success
    # Should be called only once for regular tasks
    assert evaluator.run_lm_eval_multi_gpu.call_count == 1

    # Check the call
    calls = evaluator.run_lm_eval_multi_gpu.call_args_list
    # Only call should be regular tasks without mmlu
    assert "arc_easy" in calls[0][1]["tasks"]
    assert "mmlu" not in calls[0][1]["tasks"]


def test_evaluator_cleanup():
    """
    Test evaluator cleanup method
    """
    evaluator = MagicMock()
    evaluator.model = MagicMock()
    evaluator.model.to = MagicMock(return_value=evaluator.model)

    from neurons.evaluator import Evaluator

    # Manually call cleanup
    Evaluator.cleanup(evaluator)

    # Should move model to CPU and clear cache
    evaluator.model.to.assert_called_with("cpu")


def test_pause_ddp_for_lm_eval_context():
    """
    Test pause_ddp_for_lm_eval context manager
    """
    from pathlib import Path

    from neurons.evaluator import pause_ddp_for_lm_eval

    with patch("tplr.distributed.dist_helper.is_distributed", return_value=True):
        with patch("tplr.distributed.dist_helper.safe_barrier") as mock_barrier:
            with patch("tplr.distributed.dist_helper.is_master", return_value=True):
                with pause_ddp_for_lm_eval("test") as sentinel:
                    assert isinstance(sentinel, Path)
                    assert "test" in str(sentinel)

                    # Create sentinel file
                    sentinel.touch()

                # Should have called barrier twice (enter and exit)
                assert mock_barrier.call_count == 2


def test_parse_lm_eval_results(evaluator):
    """
    Test parsing of lm-eval results from JSON output
    """
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create mock results file
        results_data = {
            "results": {
                "arc_easy": {
                    "acc,none": 0.75,
                    "acc_norm,none": 0.80,
                },
                "hellaswag": {
                    "acc,none": 0.65,
                },
                "missing_metric_task": {
                    "other_metric": 0.99,
                },
            }
        }

        results_file = output_dir / "results.json"
        results_file.write_text(json.dumps(results_data))

        # Parse results
        results = evaluator.parse_lm_eval_results(
            output_dir, window=100, global_step=50
        )

        # Check parsed results
        assert "arc_easy" in results
        assert results["arc_easy"] == 0.80  # Should prefer acc_norm over acc
        assert "hellaswag" in results
        assert results["hellaswag"] == 0.65
        assert "missing_metric_task" not in results  # No valid metric

        # Check that metrics were logged
        assert evaluator.metrics_logger.log.call_count >= 3  # At least 3 log calls
        assert evaluator.wandb.log.call_count >= 3


def test_run_custom_eval_distributed(evaluator):
    """
    Test custom evaluation with distributed reduce operations
    """
    import os

    evaluator.config.custom_eval_path = "/tmp/eval_data"
    evaluator.config.batch_size = 2
    evaluator.tokenizer = MagicMock()
    evaluator.tokenizer.pad_token_id = 0
    evaluator.tokenizer.eos_token_id = 0
    evaluator.tokenizer.batch_decode = MagicMock(return_value=["test text"] * 2)

    # Mock environment
    with patch.dict(os.environ, {"DATASET_BINS_PATH": "/tmp/eval_data"}):
        # Mock dataset
        with patch("tplr.SharedShardedDataset") as mock_dataset_class:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            # Create a mock item that returns a tensor
            mock_dataset.__getitem__ = MagicMock(
                return_value=torch.ones(128, dtype=torch.long)
            )
            mock_dataset_class.return_value = mock_dataset

            # Mock Path.exists to return True for dataset files
            with patch("pathlib.Path.exists", return_value=True):
                with patch("torch.utils.data.SubsetRandomSampler") as mock_sampler:
                    mock_sampler.return_value = [0, 1]  # Sample indices

                    with patch("torch.utils.data.DataLoader") as mock_dataloader_class:
                        # Create actual tensor batches
                        mock_batch = torch.ones(2, 128, dtype=torch.long)
                        mock_dataloader = MagicMock()
                        mock_dataloader.__iter__ = MagicMock(
                            return_value=iter([mock_batch])
                        )
                        mock_dataloader_class.return_value = mock_dataloader

                        with patch(
                            "tplr.distributed.dist_helper.ddp_reduce"
                        ) as mock_reduce:
                            mock_reduce.side_effect = (
                                lambda x, **kwargs: x * 2
                            )  # Simulate reduction

                            with patch(
                                "torchtitan.components.loss.cross_entropy_loss"
                            ) as mock_loss:
                                mock_loss.return_value = torch.tensor(2.0)

                                # Mock model output
                                evaluator.model.eval = MagicMock()
                                evaluator.model.return_value = torch.randn(
                                    2, 128, 50000
                                )

                                perplexity, loss = evaluator.run_custom_eval(100, 50)

                                assert perplexity is not None
                                assert loss is not None

                                # Check that distributed reduce was called
                                assert (
                                    mock_reduce.call_count == 3
                                )  # loss, tokens, bytes


# ============================================================================
# BACKFILL AND VERSIONING TESTS
# ============================================================================


def test_model_cache_with_namespace():
    """Test model cache with version namespace."""
    import tempfile
    from pathlib import Path

    from neurons.evaluator import ModelCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Test without namespace (backward compatibility)
        cache_no_ns = ModelCache(base_dir=cache_dir)
        assert cache_no_ns.base_dir == cache_dir
        assert cache_no_ns.get_path(100) == cache_dir / "window_100"

        # Test with namespace
        cache_v1 = ModelCache(base_dir=cache_dir, namespace="v2.1.0")
        expected_dir = cache_dir / "v2.1.0"
        assert cache_v1.base_dir == expected_dir
        assert expected_dir.exists()
        assert cache_v1.get_path(100) == expected_dir / "window_100"

        # Test different versions have isolated caches
        cache_v2 = ModelCache(base_dir=cache_dir, namespace="v3.0.0")
        path_v1 = cache_v1.get_path(100)
        path_v2 = cache_v2.get_path(100)
        assert path_v1 != path_v2
        assert path_v1 == cache_dir / "v2.1.0" / "window_100"
        assert path_v2 == cache_dir / "v3.0.0" / "window_100"


@pytest.fixture
def evaluator_with_backfill():
    """Fixture for evaluator with backfill features enabled."""
    for gen in setup_evaluator_with_mocks():
        evaluator = gen

        # Add backfill-specific attributes
        evaluator.config.from_window = None
        evaluator.config.to_window = None
        evaluator.config.no_follow = False
        evaluator.config.force_reval = False

        evaluator.from_window = evaluator.config.from_window
        evaluator.to_window = evaluator.config.to_window
        evaluator.no_follow = evaluator.config.no_follow
        evaluator.force_reval = evaluator.config.force_reval
        evaluator.tasks = []  # Add tasks attribute for evaluate_window

        # Import and bind the actual methods from Evaluator for backfill testing
        from neurons.evaluator import Evaluator

        evaluator._resolve_to_window = Evaluator._resolve_to_window.__get__(evaluator)
        evaluator.backfill_range = Evaluator.backfill_range.__get__(evaluator)

        yield evaluator


@pytest.mark.asyncio
async def test_force_reval_flag(evaluator_with_backfill):
    """Test that force_reval allows re-evaluation of already evaluated windows."""
    evaluator = evaluator_with_backfill
    evaluator.evaluated_windows.add(100)
    evaluator.start_window = 0

    # Test without force_reval
    evaluator.force_reval = False
    result = await evaluator.evaluate_window(100)
    assert result == True  # Returns True but doesn't actually evaluate

    # Test with force_reval
    evaluator.force_reval = True
    evaluator.ckpt.download_and_load = AsyncMock(return_value=(100, 50))
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model")
    evaluator.run_custom_eval = MagicMock(return_value=(5.2, 1.8))
    evaluator.model_cache = MagicMock()
    evaluator.tasks = []

    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        result = await evaluator.evaluate_window(100)

    # Should have called save_model_for_eval despite window being in evaluated_windows
    evaluator.save_model_for_eval.assert_called_once_with(100)
    assert result == True


@pytest.mark.asyncio
async def test_negative_global_step_clamping(evaluator_with_backfill):
    """Test that negative global_step values are clamped to 0."""
    evaluator = evaluator_with_backfill
    evaluator.start_window = 150  # High start window

    # Mock checkpoint at window 100 (before start_window)
    evaluator.ckpt.download_and_load = AsyncMock(
        return_value=(100, -1)
    )  # -1 means no global_step
    evaluator.save_model_for_eval = MagicMock(return_value="/tmp/model")
    evaluator.model_cache = MagicMock()
    evaluator.tasks = []

    # Track the global_step passed to run_custom_eval
    captured_step = None

    def capture_step(window, global_step):
        nonlocal captured_step
        captured_step = global_step
        return None, None

    evaluator.run_custom_eval = MagicMock(side_effect=capture_step)

    with patch("neurons.evaluator.pause_ddp_for_lm_eval"):
        await evaluator.evaluate_window(100)

    # Global step should be clamped to 0, not negative (100 - 150 would be -50)
    assert captured_step == 0


@pytest.mark.asyncio
async def test_resolve_to_window(evaluator_with_backfill):
    """Test _resolve_to_window discovers latest READY window."""
    evaluator = evaluator_with_backfill

    # Mock discovering latest window
    evaluator.ckpt._discover_latest = AsyncMock(return_value=250)
    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=True)

    with patch("tplr.distributed.dist_helper.broadcast") as mock_broadcast:
        result = await evaluator._resolve_to_window()

    assert result == 250
    evaluator.ckpt._discover_latest.assert_called_once_with(prefer_highest_staked=True)
    evaluator.ckpt.check_checkpoint_exists.assert_called_once_with(window=250)


@pytest.mark.asyncio
async def test_backfill_range_execution(evaluator_with_backfill):
    """Test executing a backfill range."""
    evaluator = evaluator_with_backfill
    evaluator.from_window = 100
    evaluator.to_window = 102
    evaluator.start_window = 0

    # Mock checkpoint checking - 101 is missing
    async def mock_check_exists(window):
        return window in [100, 102]  # 101 is not ready

    evaluator.ckpt.check_checkpoint_exists = mock_check_exists

    # Track evaluated windows
    evaluated = []

    async def mock_evaluate(window, is_baseline=False):
        evaluated.append(window)
        return True

    evaluator.evaluate_window = mock_evaluate

    with patch("tplr.distributed.dist_helper.broadcast") as mock_broadcast:
        with patch("tplr.distributed.dist_helper.safe_barrier") as mock_barrier:
            await evaluator.backfill_range()

    # Should have evaluated only windows 100 and 102 (101 was missing)
    assert evaluated == [100, 102]
    # Should have called barrier for each evaluated window
    assert mock_barrier.call_count == 2


@pytest.mark.asyncio
async def test_backfill_range_with_force_reval(evaluator_with_backfill):
    """Test backfill with force_reval re-evaluates windows."""
    evaluator = evaluator_with_backfill
    evaluator.from_window = 100
    evaluator.to_window = 101
    evaluator.force_reval = True
    evaluator.start_window = 0
    evaluator.evaluated_windows = {100, 101}  # Already evaluated

    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=True)

    # Track evaluated windows
    evaluated = []

    async def mock_evaluate(window, is_baseline=False):
        evaluated.append(window)
        return True

    evaluator.evaluate_window = mock_evaluate

    with patch("tplr.distributed.dist_helper.broadcast"):
        with patch("tplr.distributed.dist_helper.safe_barrier"):
            await evaluator.backfill_range()

    # Should have re-evaluated both windows despite them being in evaluated_windows
    assert evaluated == [100, 101]


@pytest.mark.asyncio
async def test_backfill_range_resolve_bounds(evaluator_with_backfill):
    """Test backfill resolves bounds when not fully specified."""
    evaluator = evaluator_with_backfill
    evaluator.from_window = 100
    evaluator.to_window = None  # Should resolve to latest
    evaluator.start_window = 50

    # Mock resolving to_window
    evaluator._resolve_to_window = AsyncMock(return_value=150)
    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=True)
    evaluator.evaluate_window = AsyncMock(return_value=True)

    with patch("tplr.distributed.dist_helper.broadcast") as mock_broadcast:
        # Mock broadcast to return the resolved bounds
        def broadcast_side_effect(tensor, src=0):
            if tensor.shape[0] == 2:  # bounds tensor
                tensor[0] = 100
                tensor[1] = 150

        mock_broadcast.side_effect = broadcast_side_effect

        with patch("tplr.distributed.dist_helper.safe_barrier"):
            await evaluator.backfill_range()

    # Should have called _resolve_to_window since to_window was None
    evaluator._resolve_to_window.assert_called_once()

    # Should evaluate all windows in range (mocked to always be ready)
    assert evaluator.evaluate_window.call_count == 51  # 100 to 150 inclusive


@pytest.mark.asyncio
async def test_run_with_no_follow(evaluator_with_backfill):
    """Test that no_follow flag prevents entering the main loop."""
    evaluator = evaluator_with_backfill
    evaluator.from_window = 100
    evaluator.to_window = 100
    evaluator.no_follow = True

    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.check_latest_checkpoint = AsyncMock(return_value=None)
    # Mock backfill_range for this test
    evaluator.backfill_range = AsyncMock()

    # Run should return early due to no_follow
    await evaluator.run()

    # backfill_range should have been called
    evaluator.backfill_range.assert_called_once()


@pytest.mark.asyncio
async def test_run_backfill_mode(evaluator_with_backfill):
    """Test run enters backfill mode when from_window or to_window is set."""
    evaluator = evaluator_with_backfill
    evaluator.from_window = 100
    evaluator.to_window = 105
    evaluator.no_follow = False  # Will continue to main loop

    evaluator.comms.get_commitments = AsyncMock(return_value={})
    evaluator.comms.get_start_window = AsyncMock(return_value=0)
    evaluator.check_latest_checkpoint = AsyncMock(
        side_effect=[None, KeyboardInterrupt()]
    )
    evaluator.backfill_range = AsyncMock()

    with patch("asyncio.sleep", new_callable=AsyncMock):
        try:
            await evaluator.run()
        except KeyboardInterrupt:
            pass

    # Should have called backfill_range before entering main loop
    evaluator.backfill_range.assert_called_once()


@pytest.mark.asyncio
async def test_backfill_flipped_bounds(evaluator_with_backfill):
    """Test backfill handles flipped bounds correctly."""
    evaluator = evaluator_with_backfill
    evaluator.from_window = 150  # Higher than to_window
    evaluator.to_window = 100  # Lower than from_window
    evaluator.start_window = 0

    evaluator.ckpt.check_checkpoint_exists = AsyncMock(return_value=True)

    # Track evaluated windows
    evaluated = []

    async def mock_evaluate(window, is_baseline=False):
        evaluated.append(window)
        return True

    evaluator.evaluate_window = mock_evaluate

    with patch("tplr.distributed.dist_helper.broadcast") as mock_broadcast:
        # Mock broadcast to return the flipped bounds
        def broadcast_side_effect(tensor, src=0):
            if tensor.shape[0] == 2:  # bounds tensor
                tensor[0] = 150
                tensor[1] = 100

        mock_broadcast.side_effect = broadcast_side_effect

        with patch("tplr.distributed.dist_helper.safe_barrier"):
            await evaluator.backfill_range()

    # Should evaluate from 100 to 150 (normalized order)
    assert min(evaluated) == 100
    assert max(evaluated) == 150
    assert len(evaluated) == 51  # 100 to 150 inclusive


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
