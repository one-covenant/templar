# Robust Shard Downloads (Issue #600)

## Overview

This document describes the improvements made to shard download reliability in the templar codebase, addressing [Issue #600](https://github.com/one-covenant/templar/issues/600).

## Problem Statement

The previous implementation of shard downloads had several reliability issues:

1. **No retry logic**: Downloads that failed or stalled would not automatically retry
2. **No validation**: System didn't verify files were successfully downloaded before proceeding
3. **Silent failures**: Download failures could leave the system in a broken state
4. **Background preparation risks**: Shards prepared in background could fail without detection until swap time

## Solution

The improved implementation adds robust error handling with three key components:

### 1. Retry Logic with Exponential Backoff

Downloads now automatically retry on failure with exponential backoff:

```python
for attempt in range(max_download_retries):  # Default: 3 attempts
    try:
        success = await download_with_timeout()
        if success:
            return True
    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_download_retries - 1:
            backoff_time = 2 ** attempt  # 1s, 2s, 4s, etc.
            await asyncio.sleep(backoff_time)
```

### 2. File Validation

All shard files are validated before and after download:

- **Existence check**: Verifies both tokens and IDs files exist
- **Size check**: Ensures files have non-zero size
- **Post-download validation**: Confirms successful download before proceeding

```python
def _validate_shard_files(tokens_file, ids_file) -> bool:
    if not os.path.exists(tokens_file) or not os.path.exists(ids_file):
        return False
    
    tokens_size = os.path.getsize(tokens_file)
    ids_size = os.path.getsize(ids_file)
    
    if tokens_size == 0 or ids_size == 0:
        return False
    
    return True
```

### 3. Await-Based Direct Download

The system now uses an await-based approach to ensure downloads complete before proceeding:

```python
async def create_dataset(shard_index: int):
    if rank == 0:
        download_task = prepare_shard(shard_index)
        success = await download_task  # Block until download completes
        
        if not success:
            raise RuntimeError("Failed to download shard")
    
    # Validate files exist before creating dataset
    if not validate_shard_files(...):
        raise RuntimeError("Shard files invalid")
    
    return SharedShardedDataset(...)
```

## Key Features

### Configurable Parameters

The `ShardedDatasetManager` now accepts configuration parameters:

```python
manager = ShardedDatasetManager(
    sequence_length=2048,
    rank=0,
    world_size=1,
    comms=comms,
    max_download_retries=3,      # Number of retry attempts
    download_timeout=600,         # Timeout per attempt (seconds)
)
```

### Comprehensive Error Logging

All failure modes are logged with clear, actionable messages:

```plaintext
ERROR: Timeout downloading shard 5 after 600s (attempt 2/3)
INFO: Retrying in 2s...
ERROR: Failed to download shard 5 after 3 attempts. Files: /path/to/train_000005.npy, /path/to/sample_ids_000005.npy
```

### Graceful Failure Handling

The system implements multiple levels of fallback:

1. **Background preparation fails**: Retry synchronously during swap
2. **Synchronous retry fails**: Raise clear error with file paths
3. **Validation fails**: Prevent dataset creation with broken state

```python
async def swap_datasets():
    success = await upcoming_dataset
    if not success:
        logger.warning("Background preparation failed. Attempting synchronous download...")
        success = await prepare_shard_with_retry(...)
        if not success:
            raise RuntimeError("Cannot swap to invalid dataset")
```

## Usage

### Basic Usage

No changes required for existing code. The improvements are transparent:

```python
# Existing code works as before
manager = ShardedDatasetManager(
    sequence_length=2048,
    rank=0,
    world_size=1,
    comms=comms,
)

# Initialize with first shard
await manager.initialize_datasets(current_shard_index=0)

# Swap to next shard (now with robust error handling)
new_shard = await manager.swap_datasets()
```

### Custom Configuration

For environments with unreliable networks, adjust retry and timeout settings:

```python
manager = ShardedDatasetManager(
    sequence_length=2048,
    rank=0,
    world_size=1,
    comms=comms,
    max_download_retries=5,      # More retries for unreliable networks
    download_timeout=1200,        # Longer timeout for slow connections
)
```

## Acceptance Criteria Met

✅ **Shard downloads succeed reliably**: Retry logic with exponential backoff handles transient failures

✅ **No broken state**: Validation ensures system never proceeds with invalid data

✅ **Graceful failure with clear logs**: All failure modes logged with actionable error messages

✅ **Await-based direct download**: System blocks on critical downloads to ensure completion

✅ **Reasonable timeframe**: Configurable timeout (default 600s = 10 minutes per attempt)

## Testing

Comprehensive unit tests cover all scenarios:

- ✅ Validation of existing files
- ✅ Successful download on first attempt
- ✅ Retry on transient failures
- ✅ Exhausted retries handling
- ✅ Timeout handling
- ✅ Background download failure with successful retry
- ✅ Complete failure with clear error

Run tests:

```bash
pytest tests/unit/test_robust_shard_download.py -v
```

## Migration Guide

### For Existing Deployments

No code changes required. The improvements are backward compatible.

### For New Deployments

Consider setting custom parameters based on your environment:

**High-bandwidth, reliable network:**
```python
max_download_retries=2
download_timeout=300  # 5 minutes
```

**Low-bandwidth or unreliable network:**
```python
max_download_retries=5
download_timeout=1800  # 30 minutes
```

## Performance Impact

- **Minimal overhead**: Validation adds <1s per shard
- **Improved reliability**: Reduces failed training runs
- **Better observability**: Clear logging helps diagnose issues

## Future Improvements

Potential enhancements for future iterations:

1. **Partial download resume**: Resume interrupted downloads from last checkpoint
2. **Parallel validation**: Validate files while downloading next shard
3. **Health metrics**: Track download success rates and retry statistics
4. **Adaptive timeouts**: Adjust timeouts based on observed download speeds

## Related Files

- `src/tplr/sharded_dataset.py`: Main implementation
- `tests/unit/test_robust_shard_download.py`: Unit tests
- `docs/shared_sharded_dataset.md`: General dataset documentation

## References

- [Issue #600](https://github.com/one-covenant/templar/issues/600): Original issue
- [PR #663](https://github.com/one-covenant/templar/pull/663): Implementation PR
