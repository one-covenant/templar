"""
Tests for checkpoint fallback and catchup functions in src/tplr/neurons.py
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

# Import the functions we're testing
from src.tplr.neurons import handle_checkpoint_catchup, load_checkpoint_with_fallback


@pytest.fixture
def mock_instance():
    """Create a mock miner/validator instance for testing"""
    instance = MagicMock()

    # Setup basic attributes
    instance.uid = 123
    instance.start_window = 100
    instance.current_window = 150
    instance.global_step = 0
    instance.model_initialized = False
    instance.inner_scheduler_step_count = 0

    # Setup hparams
    instance.hparams = MagicMock()
    instance.hparams.inner_steps = 30
    instance.hparams.checkpoint_init_window = None

    # Setup checkpoint manager
    instance.ckpt = MagicMock()
    instance.ckpt._discover_latest = AsyncMock()
    instance.ckpt.download_and_load = AsyncMock()

    # Setup model
    instance.model = MagicMock()

    # Setup comms
    instance.comms = MagicMock()

    # Bootstrap version (can be overridden in tests)
    instance.bootstrap_version = None

    # Mock should_skip_scheduler_step to return False (no flattening by default)
    instance.should_skip_scheduler_step = MagicMock(return_value=False)

    return instance


class TestLoadCheckpointWithFallback:
    """Test suite for load_checkpoint_with_fallback function"""

    @pytest.mark.asyncio
    async def test_load_from_current_version(self, mock_instance):
        """Test loading checkpoint from current version when available"""
        # Setup: current version has checkpoint at window 120
        mock_instance.ckpt._discover_latest.return_value = 120
        mock_instance.ckpt.download_and_load.return_value = (
            120,
            20,
        )  # (window, global_step)

        # Execute
        (
            ckpt_ok,
            ckpt_sync_win,
            ckpt_global_step,
            from_bootstrap,
        ) = await load_checkpoint_with_fallback(mock_instance)

        # Verify
        assert ckpt_ok is True
        assert ckpt_sync_win == 120
        assert ckpt_global_step == 20
        assert from_bootstrap is False
        assert mock_instance.model_initialized is True
        assert mock_instance.global_step == 20

        # Should only check current version
        mock_instance.ckpt._discover_latest.assert_called_once_with(
            prefer_highest_staked=True
        )
        mock_instance.ckpt.download_and_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_bootstrap_version(self, mock_instance):
        """Test falling back to bootstrap version when no current version exists"""
        # Setup: no current version checkpoint, bootstrap configured
        mock_instance.ckpt._discover_latest.return_value = None
        mock_instance.bootstrap_version = "v1.0.0"

        # Create mock bootstrap checkpoint manager
        with patch("src.tplr.neurons.tplr.DCPCheckpointer") as mock_dcp:
            bootstrap_ckpt = MagicMock()
            bootstrap_ckpt._discover_latest = AsyncMock(return_value=110)
            bootstrap_ckpt.download_and_load = AsyncMock(return_value=(110, 10))
            mock_dcp.return_value = bootstrap_ckpt

            # Execute
            (
                ckpt_ok,
                ckpt_sync_win,
                ckpt_global_step,
                from_bootstrap,
            ) = await load_checkpoint_with_fallback(mock_instance)

        # Verify
        assert ckpt_ok is True
        assert ckpt_sync_win == 110
        assert ckpt_global_step == 10
        assert from_bootstrap is True
        assert mock_instance.model_initialized is True
        assert mock_instance.global_step == 10

        # Should have tried bootstrap after current version failed
        mock_dcp.assert_called_once_with(
            mock_instance.comms, uid=123, version="v1.0.0", repo_root="."
        )

    @pytest.mark.asyncio
    async def test_use_specific_bootstrap_window(self, mock_instance):
        """Test using checkpoint_init_window to load specific bootstrap checkpoint"""
        # Setup: no current version, bootstrap with specific window configured
        mock_instance.ckpt._discover_latest.return_value = None
        mock_instance.bootstrap_version = "v1.0.0"
        mock_instance.hparams.checkpoint_init_window = 105  # Specific window

        # Create mock bootstrap checkpoint manager
        with patch("src.tplr.neurons.tplr.DCPCheckpointer") as mock_dcp:
            bootstrap_ckpt = MagicMock()
            # Should not call _discover_latest since specific window is set
            bootstrap_ckpt.download_and_load = AsyncMock(return_value=(105, 5))
            mock_dcp.return_value = bootstrap_ckpt

            # Execute
            (
                ckpt_ok,
                ckpt_sync_win,
                ckpt_global_step,
                from_bootstrap,
            ) = await load_checkpoint_with_fallback(mock_instance)

        # Verify
        assert ckpt_ok is True
        assert ckpt_sync_win == 105
        assert ckpt_global_step == 5
        assert from_bootstrap is True

        # Should use specific window, not discover latest
        bootstrap_ckpt.download_and_load.assert_called_once_with(
            model=mock_instance.model,
            window=105,
            shared_fs=True,
            process_group=None,
            prefer_highest_staked=True,
        )

    @pytest.mark.asyncio
    async def test_calculate_global_step_when_missing(self, mock_instance):
        """Test global_step calculation when checkpoint returns -1"""
        # Setup: checkpoint has no global_step stored (returns -1)
        mock_instance.ckpt._discover_latest.return_value = 120
        mock_instance.ckpt.download_and_load.return_value = (
            120,
            -1,
        )  # -1 means no global_step
        mock_instance.start_window = 100

        # Execute
        (
            ckpt_ok,
            ckpt_sync_win,
            ckpt_global_step,
            from_bootstrap,
        ) = await load_checkpoint_with_fallback(mock_instance)

        # Verify - should calculate: 120 - 100 = 20
        assert ckpt_ok is True
        assert ckpt_sync_win == 120
        assert ckpt_global_step == 20  # Calculated from window difference
        assert mock_instance.global_step == 20

    @pytest.mark.asyncio
    async def test_no_checkpoint_available(self, mock_instance):
        """Test when no checkpoint is available at all"""
        # Setup: no current version, no bootstrap configured
        mock_instance.ckpt._discover_latest.return_value = None
        mock_instance.bootstrap_version = None

        # Execute
        (
            ckpt_ok,
            ckpt_sync_win,
            ckpt_global_step,
            from_bootstrap,
        ) = await load_checkpoint_with_fallback(mock_instance)

        # Verify
        assert ckpt_ok is False
        assert ckpt_sync_win == 0
        assert ckpt_global_step == 0
        assert from_bootstrap is False
        assert mock_instance.model_initialized is False  # Should remain False

    @pytest.mark.asyncio
    async def test_bootstrap_checkpoint_not_found(self, mock_instance):
        """Test when bootstrap version is configured but no checkpoint exists"""
        # Setup: no current version, bootstrap configured but no checkpoint
        mock_instance.ckpt._discover_latest.return_value = None
        mock_instance.bootstrap_version = "v1.0.0"

        with patch("src.tplr.neurons.tplr.DCPCheckpointer") as mock_dcp:
            bootstrap_ckpt = MagicMock()
            bootstrap_ckpt._discover_latest = AsyncMock(
                return_value=None
            )  # No checkpoint
            mock_dcp.return_value = bootstrap_ckpt

            # Execute
            (
                ckpt_ok,
                ckpt_sync_win,
                ckpt_global_step,
                from_bootstrap,
            ) = await load_checkpoint_with_fallback(mock_instance)

        # Verify
        assert ckpt_ok is False
        assert ckpt_sync_win == 0
        assert ckpt_global_step == 0
        assert from_bootstrap is False
        assert mock_instance.model_initialized is False


class TestHandleCheckpointCatchup:
    """Test suite for handle_checkpoint_catchup function"""

    @pytest.mark.asyncio
    async def test_no_checkpoint_catchup_from_start(self, mock_instance):
        """Test catchup from start_window when no checkpoint loaded"""
        # Mock the catchup function
        with patch("src.tplr.neurons.catchup_with_aggregation_server") as mock_catchup:
            mock_catchup.return_value = None

            # Execute - no checkpoint loaded
            await handle_checkpoint_catchup(
                mock_instance,
                ckpt_ok=False,
                ckpt_sync_win=0,
                ckpt_global_step=0,
                from_bootstrap=False,
            )

            # Verify - should catch up from start_window
            mock_catchup.assert_called_once_with(
                mock_instance, mock_instance.start_window, aggregator_device=None
            )

            # No scheduler steps to replay when no checkpoint
            assert mock_instance.inner_scheduler.step.call_count == 0

    @pytest.mark.asyncio
    async def test_bootstrap_catchup_from_start(self, mock_instance):
        """Test catchup from start_window when loaded from bootstrap"""
        mock_instance.inner_scheduler = MagicMock()

        # Mock the catchup function
        with patch("src.tplr.neurons.catchup_with_aggregation_server") as mock_catchup:
            mock_catchup.return_value = None

            # Execute - loaded from bootstrap
            await handle_checkpoint_catchup(
                mock_instance,
                ckpt_ok=True,
                ckpt_sync_win=110,
                ckpt_global_step=10,
                from_bootstrap=True,
            )

            # Verify - should catch up from start_window (not checkpoint window)
            mock_catchup.assert_called_once_with(
                mock_instance, mock_instance.start_window, aggregator_device=None
            )

            # Should replay scheduler steps: 10 * 30 = 300
            assert mock_instance.inner_scheduler.step.call_count == 300

    @pytest.mark.asyncio
    async def test_current_checkpoint_behind_catchup(self, mock_instance):
        """Test catchup when current version checkpoint is behind current window"""
        mock_instance.inner_scheduler = MagicMock()
        mock_instance.current_window = 150
        mock_instance.start_window = 100

        # Mock the catchup function
        with patch("src.tplr.neurons.catchup_with_aggregation_server") as mock_catchup:
            mock_catchup.return_value = None

            # Execute - checkpoint at 120, current at 150
            await handle_checkpoint_catchup(
                mock_instance,
                ckpt_ok=True,
                ckpt_sync_win=120,
                ckpt_global_step=20,
                from_bootstrap=False,
            )

            # Verify - should catch up from checkpoint window (120)
            mock_catchup.assert_called_once_with(
                mock_instance, 120, aggregator_device=None
            )

            # Should replay scheduler steps: 20 * 30 = 600
            assert mock_instance.inner_scheduler.step.call_count == 600

    @pytest.mark.asyncio
    async def test_checkpoint_up_to_date_no_catchup(self, mock_instance):
        """Test no catchup when checkpoint is up to date"""
        mock_instance.inner_scheduler = MagicMock()
        mock_instance.current_window = 120

        # Mock the catchup function
        with patch("src.tplr.neurons.catchup_with_aggregation_server") as mock_catchup:
            mock_catchup.return_value = None

            # Execute - checkpoint at current window
            await handle_checkpoint_catchup(
                mock_instance,
                ckpt_ok=True,
                ckpt_sync_win=120,
                ckpt_global_step=20,
                from_bootstrap=False,
            )

            # Verify - should not catch up
            mock_catchup.assert_not_called()

            # Should still replay scheduler steps: 20 * 30 = 600
            assert mock_instance.inner_scheduler.step.call_count == 600

    @pytest.mark.asyncio
    async def test_catchup_respects_start_window_boundary(self, mock_instance):
        """Test catchup uses max(checkpoint_window, start_window)"""
        mock_instance.inner_scheduler = MagicMock()
        mock_instance.current_window = 150
        mock_instance.start_window = 100

        # Mock the catchup function
        with patch("src.tplr.neurons.catchup_with_aggregation_server") as mock_catchup:
            mock_catchup.return_value = None

            # Execute - checkpoint at 90 (before start_window)
            await handle_checkpoint_catchup(
                mock_instance,
                ckpt_ok=True,
                ckpt_sync_win=90,  # Before start_window
                ckpt_global_step=5,
                from_bootstrap=False,
            )

            # Verify - should catch up from start_window (100), not checkpoint (90)
            mock_catchup.assert_called_once_with(
                mock_instance, 100, aggregator_device=None
            )

            # Should replay scheduler steps: 5 * 30 = 150
            assert mock_instance.inner_scheduler.step.call_count == 150

    @pytest.mark.asyncio
    async def test_no_scheduler_replay_when_global_step_zero(self, mock_instance):
        """Test no scheduler replay when global_step is 0"""
        mock_instance.inner_scheduler = MagicMock()

        with patch("src.tplr.neurons.catchup_with_aggregation_server") as mock_catchup:
            mock_catchup.return_value = None

            # Execute - global_step is 0
            await handle_checkpoint_catchup(
                mock_instance,
                ckpt_ok=True,
                ckpt_sync_win=100,
                ckpt_global_step=0,
                from_bootstrap=False,
            )

            # Should not replay any scheduler steps
            assert mock_instance.inner_scheduler.step.call_count == 0
