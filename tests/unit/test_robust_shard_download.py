"""
Unit tests for robust shard download functionality (Issue #600).

Tests the improved download logic with retry, validation, and await-based approach.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import torch.distributed as dist

import tplr
from tplr.sharded_dataset import ShardedDatasetManager, SharedShardedDataset


@pytest.fixture
def mock_comms():
    """Create a mock Comms object."""
    comms = MagicMock()
    comms.get_own_bucket = MagicMock(return_value=MagicMock(name="test-bucket"))
    comms.s3_get_object = AsyncMock()
    return comms


@pytest.fixture
def temp_dataset_path(tmp_path):
    """Create a temporary dataset path."""
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    return dataset_path


@pytest.fixture
def dataset_manager(mock_comms, temp_dataset_path, monkeypatch):
    """Create a ShardedDatasetManager instance for testing."""
    monkeypatch.setenv("DATASET_BINS_PATH", str(temp_dataset_path))
    manager = ShardedDatasetManager(
        sequence_length=2048,
        rank=0,
        world_size=1,
        comms=mock_comms,
        max_download_retries=3,
        download_timeout=60,
    )
    return manager


class TestShardValidation:
    """Tests for shard file validation."""

    def test_validate_shard_files_both_exist(self, dataset_manager, temp_dataset_path):
        """Test validation when both files exist with non-zero size."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Create files with content
        np.save(tokens_file, np.array([1, 2, 3], dtype=np.uint32))
        np.save(ids_file, np.array([100, 200, 300], dtype=np.uint64))

        assert dataset_manager._validate_shard_files(tokens_file, ids_file) is True

    def test_validate_shard_files_missing_tokens(
        self, dataset_manager, temp_dataset_path
    ):
        """Test validation when tokens file is missing."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Only create ids file
        np.save(ids_file, np.array([100, 200, 300], dtype=np.uint64))

        assert dataset_manager._validate_shard_files(tokens_file, ids_file) is False

    def test_validate_shard_files_missing_ids(self, dataset_manager, temp_dataset_path):
        """Test validation when IDs file is missing."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Only create tokens file
        np.save(tokens_file, np.array([1, 2, 3], dtype=np.uint32))

        assert dataset_manager._validate_shard_files(tokens_file, ids_file) is False

    def test_validate_shard_files_zero_size(self, dataset_manager, temp_dataset_path):
        """Test validation when files have zero size."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Create empty files
        tokens_file.touch()
        ids_file.touch()

        assert dataset_manager._validate_shard_files(tokens_file, ids_file) is False


class TestDownloadWithValidation:
    """Tests for download with validation."""

    @pytest.mark.asyncio
    async def test_download_success(self, dataset_manager, temp_dataset_path):
        """Test successful download with validation."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Mock successful downloads that create files
        async def mock_download(key, bucket, load_data=False, show_progress=True):
            # Create the file
            if "train_" in str(key):
                np.save(key, np.array([1, 2, 3], dtype=np.uint32))
            else:
                np.save(key, np.array([100, 200, 300], dtype=np.uint64))
            return str(key)

        dataset_manager.comms.s3_get_object = AsyncMock(side_effect=mock_download)

        bucket = dataset_manager.comms.get_own_bucket("dataset", "read")
        success = await dataset_manager._download_files_with_validation(
            bucket, tokens_file, ids_file, shard_index=0
        )

        assert success is True
        assert os.path.exists(tokens_file)
        assert os.path.exists(ids_file)

    @pytest.mark.asyncio
    async def test_download_returns_none(self, dataset_manager, temp_dataset_path):
        """Test handling when download returns None."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Mock download that returns None
        dataset_manager.comms.s3_get_object = AsyncMock(return_value=None)

        bucket = dataset_manager.comms.get_own_bucket("dataset", "read")
        success = await dataset_manager._download_files_with_validation(
            bucket, tokens_file, ids_file, shard_index=0
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_download_raises_exception(self, dataset_manager, temp_dataset_path):
        """Test handling when download raises an exception."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Mock download that raises exception
        dataset_manager.comms.s3_get_object = AsyncMock(
            side_effect=Exception("Network error")
        )

        bucket = dataset_manager.comms.get_own_bucket("dataset", "read")
        success = await dataset_manager._download_files_with_validation(
            bucket, tokens_file, ids_file, shard_index=0
        )

        assert success is False


class TestPrepareShardWithRetry:
    """Tests for shard preparation with retry logic."""

    @pytest.mark.asyncio
    async def test_prepare_shard_already_exists(
        self, dataset_manager, temp_dataset_path
    ):
        """Test preparation when shard already exists."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Create valid files
        np.save(tokens_file, np.array([1, 2, 3], dtype=np.uint32))
        np.save(ids_file, np.array([100, 200, 300], dtype=np.uint64))

        success = await dataset_manager._prepare_shard_with_retry(
            shard_index=0,
            remapped_shard=0,
            tokens_file=tokens_file,
            ids_file=ids_file,
        )

        assert success is True
        # Should not call download since files exist
        dataset_manager.comms.s3_get_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_shard_download_success_first_attempt(
        self, dataset_manager, temp_dataset_path
    ):
        """Test successful download on first attempt."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Mock successful download
        async def mock_download(key, bucket, load_data=False, show_progress=True):
            if "train_" in str(key):
                np.save(key, np.array([1, 2, 3], dtype=np.uint32))
            else:
                np.save(key, np.array([100, 200, 300], dtype=np.uint64))
            return str(key)

        dataset_manager.comms.s3_get_object = AsyncMock(side_effect=mock_download)

        success = await dataset_manager._prepare_shard_with_retry(
            shard_index=0,
            remapped_shard=0,
            tokens_file=tokens_file,
            ids_file=ids_file,
        )

        assert success is True
        assert os.path.exists(tokens_file)
        assert os.path.exists(ids_file)

    @pytest.mark.asyncio
    async def test_prepare_shard_retry_on_failure(
        self, dataset_manager, temp_dataset_path
    ):
        """Test retry logic when download fails initially."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        call_count = 0

        async def mock_download_with_retry(
            key, bucket, load_data=False, show_progress=True
        ):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Fail first two attempts
                return None
            # Succeed on third attempt
            if "train_" in str(key):
                np.save(key, np.array([1, 2, 3], dtype=np.uint32))
            else:
                np.save(key, np.array([100, 200, 300], dtype=np.uint64))
            return str(key)

        dataset_manager.comms.s3_get_object = AsyncMock(
            side_effect=mock_download_with_retry
        )

        success = await dataset_manager._prepare_shard_with_retry(
            shard_index=0,
            remapped_shard=0,
            tokens_file=tokens_file,
            ids_file=ids_file,
        )

        assert success is True
        # Should have retried (2 failed attempts for each file + 1 success = 3 attempts)
        assert call_count >= 4  # At least 2 calls per file

    @pytest.mark.asyncio
    async def test_prepare_shard_all_retries_exhausted(
        self, dataset_manager, temp_dataset_path
    ):
        """Test behavior when all retry attempts are exhausted."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Mock download that always fails
        dataset_manager.comms.s3_get_object = AsyncMock(return_value=None)

        success = await dataset_manager._prepare_shard_with_retry(
            shard_index=0,
            remapped_shard=0,
            tokens_file=tokens_file,
            ids_file=ids_file,
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_prepare_shard_timeout(self, dataset_manager, temp_dataset_path):
        """Test handling of download timeout."""
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Mock download that times out
        async def slow_download(*args, **kwargs):
            await asyncio.sleep(100)  # Longer than timeout
            return "success"

        dataset_manager.comms.s3_get_object = AsyncMock(side_effect=slow_download)
        dataset_manager.download_timeout = 1  # Short timeout for testing

        success = await dataset_manager._prepare_shard_with_retry(
            shard_index=0,
            remapped_shard=0,
            tokens_file=tokens_file,
            ids_file=ids_file,
        )

        assert success is False


class TestSwapDatasetsRobustness:
    """Tests for robust dataset swapping."""

    @pytest.mark.asyncio
    async def test_swap_with_successful_background_download(
        self, dataset_manager, temp_dataset_path, monkeypatch
    ):
        """Test swap when background download succeeded."""
        # Mock successful upcoming dataset task
        dataset_manager.upcoming_dataset = asyncio.create_task(
            asyncio.coroutine(lambda: True)()
        )

        # Mock initialize_datasets
        dataset_manager.initialize_datasets = AsyncMock()
        dataset_manager.active_dataset = None

        new_shard = await dataset_manager.swap_datasets()

        assert new_shard == 1
        dataset_manager.initialize_datasets.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_swap_with_failed_background_download_retry_success(
        self, dataset_manager, temp_dataset_path, monkeypatch
    ):
        """Test swap when background download failed but retry succeeds."""
        # Mock failed upcoming dataset task
        dataset_manager.upcoming_dataset = asyncio.create_task(
            asyncio.coroutine(lambda: False)()
        )

        # Create valid shard files for retry
        tokens_file = temp_dataset_path / "train_000001.npy"
        ids_file = temp_dataset_path / "sample_ids_000001.npy"
        np.save(tokens_file, np.array([1, 2, 3], dtype=np.uint32))
        np.save(ids_file, np.array([100, 200, 300], dtype=np.uint64))

        # Mock initialize_datasets
        dataset_manager.initialize_datasets = AsyncMock()
        dataset_manager.active_dataset = None

        new_shard = await dataset_manager.swap_datasets()

        assert new_shard == 1
        dataset_manager.initialize_datasets.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_swap_with_failed_background_download_retry_fails(
        self, dataset_manager, temp_dataset_path
    ):
        """Test swap when both background download and retry fail."""
        # Mock failed upcoming dataset task
        dataset_manager.upcoming_dataset = asyncio.create_task(
            asyncio.coroutine(lambda: False)()
        )

        # Don't create files, so retry will fail
        dataset_manager.comms.s3_get_object = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="Failed to download shard"):
            await dataset_manager.swap_datasets()


class TestCreateDatasetAwaitBased:
    """Tests for await-based dataset creation."""

    @pytest.mark.asyncio
    async def test_create_dataset_success(
        self, dataset_manager, temp_dataset_path, monkeypatch
    ):
        """Test successful dataset creation with await-based download."""
        # Create valid shard files
        tokens_file = temp_dataset_path / "train_000000.npy"
        ids_file = temp_dataset_path / "sample_ids_000000.npy"

        # Create files with proper structure
        tokens = np.array([1, 2, 3, 4] * 512, dtype=np.uint32)  # 2048 tokens
        sample_ids = np.array([100], dtype=np.uint64)
        np.save(tokens_file, tokens)
        np.save(ids_file, sample_ids)

        # Mock dist.barrier for single-process test
        with patch("torch.distributed.barrier"):
            dataset = await dataset_manager.create_dataset(shard_index=0)

        assert dataset is not None
        assert isinstance(dataset, SharedShardedDataset)

    @pytest.mark.asyncio
    async def test_create_dataset_download_failure(
        self, dataset_manager, temp_dataset_path
    ):
        """Test dataset creation when download fails."""
        # Mock download that always fails
        dataset_manager.comms.s3_get_object = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="Failed to download shard"):
            await dataset_manager.create_dataset(shard_index=0)

    @pytest.mark.asyncio
    async def test_create_dataset_validation_failure(
        self, dataset_manager, temp_dataset_path
    ):
        """Test dataset creation when validation fails after download."""

        # Mock download that succeeds but creates invalid files
        async def mock_download(key, bucket, load_data=False, show_progress=True):
            # Create empty files
            Path(key).touch()
            return str(key)

        dataset_manager.comms.s3_get_object = AsyncMock(side_effect=mock_download)

        with pytest.raises(RuntimeError, match="files do not exist or are invalid"):
            await dataset_manager.create_dataset(shard_index=0)


class TestShardRemapping:
    """Tests for shard index remapping."""

    def test_remap_shard_7_to_13(self, dataset_manager):
        """Test that shard 7 is remapped to 13."""
        assert dataset_manager.remap_shard_index(7) == 13

    def test_remap_other_shards_unchanged(self, dataset_manager):
        """Test that other shard indices remain unchanged."""
        for i in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]:
            assert dataset_manager.remap_shard_index(i) == i


class TestConfigurableParameters:
    """Tests for configurable retry and timeout parameters."""

    def test_custom_retry_count(self, mock_comms, temp_dataset_path, monkeypatch):
        """Test custom max_download_retries parameter."""
        monkeypatch.setenv("DATASET_BINS_PATH", str(temp_dataset_path))
        manager = ShardedDatasetManager(
            sequence_length=2048,
            rank=0,
            world_size=1,
            comms=mock_comms,
            max_download_retries=5,
            download_timeout=60,
        )
        assert manager.max_download_retries == 5

    def test_custom_timeout(self, mock_comms, temp_dataset_path, monkeypatch):
        """Test custom download_timeout parameter."""
        monkeypatch.setenv("DATASET_BINS_PATH", str(temp_dataset_path))
        manager = ShardedDatasetManager(
            sequence_length=2048,
            rank=0,
            world_size=1,
            comms=mock_comms,
            max_download_retries=3,
            download_timeout=300,
        )
        assert manager.download_timeout == 300
