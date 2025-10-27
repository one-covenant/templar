#!/usr/bin/env python3
"""
Script to download dataset files from R2 using the efficient download method
from the templar codebase.

Usage:
    python scripts/download_dataset.py                          # Downloads shards 0 and 1 to default path
    python scripts/download_dataset.py 0 1 2 3                  # Downloads shards 0, 1, 2, and 3
    python scripts/download_dataset.py 5                        # Downloads only shard 5
    python scripts/download_dataset.py --path /custom/path 0 1  # Downloads to custom path
    python scripts/download_dataset.py -p ./data 2 3 4          # Downloads to ./data
"""

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import bittensor as bt
from dotenv import load_dotenv

# Add the parent directory to the path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import tplr
from src.tplr.comms import Comms
from src.tplr.schemas import Bucket
from src.tplr.sharded_dataset import SharedShardedDataset

# Load environment variables
load_dotenv()


async def download_files(shard_indices, output_path):
    """Download the dataset files from R2.

    Args:
        shard_indices: List of shard indices to download
        output_path: Base directory path where files will be saved (e.g., "remote/tokenized/")
    """

    # Get credentials from environment variables (using the same env vars as comms)
    account_id = os.getenv("R2_DATASET_ACCOUNT_ID")
    bucket_name = os.getenv("R2_DATASET_BUCKET_NAME")
    aws_access_key_id = os.getenv("R2_DATASET_READ_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("R2_DATASET_READ_SECRET_ACCESS_KEY")

    if not all([aws_access_key_id, aws_secret_access_key, account_id, bucket_name]):
        print("Error: Missing required environment variables.")
        print("Please set the following in your .env file:")
        print("  - R2_DATASET_ACCOUNT_ID")
        print("  - R2_DATASET_BUCKET_NAME")
        print("  - R2_DATASET_READ_ACCESS_KEY_ID")
        print("  - R2_DATASET_READ_SECRET_ACCESS_KEY")
        sys.exit(1)

    # Create bucket configuration
    bucket = Bucket(
        name=bucket_name,
        account_id=account_id,
        access_key_id=aws_access_key_id,
        secret_access_key=aws_secret_access_key,
    )

    # Load hparams
    hparams = tplr.load_hparams()

    # Create a proper bt config with finney network
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.netuid = 268
    config.subtensor.network = "finney"

    # Create a dummy wallet for Comms initialization
    dummy_wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        ),
        coldkey=SimpleNamespace(
            ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        ),
    )

    # Initialize Comms normally
    comms = Comms(
        wallet=dummy_wallet,
        config=config,
        bucket=bucket,
        hparams=hparams,
        uid=0,
    )

    # Create the output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_path.absolute()}")

    # Change to output directory so s3_get_object saves files there
    # This avoids filling up the root filesystem with temp files
    original_dir = os.getcwd()
    os.chdir(output_path)

    # Build list of files to download based on shard indices
    files_to_download = []
    skipped_files = []

    for idx in shard_indices:
        # Get just the filenames (we're now in the output directory)
        filename_tokens = f"train_{idx:06d}.npy"
        filename_ids = f"sample_ids_{idx:06d}.bin"

        # The S3 keys are under the "tokenized/" prefix
        for filename in [filename_tokens, filename_ids]:
            local_file = Path(filename)

            if local_file.exists():
                file_size = local_file.stat().st_size
                print(
                    f"✓ Skipping {filename} (already exists, size: {file_size:,} bytes)"
                )
                skipped_files.append(str(local_file))
            else:
                # Construct S3 key from the filename (add tokenized/ prefix)
                s3_key = f"tokenized/{filename}"
                # s3_get_object will save to s3_key path (tokenized/train_000000.npy)
                # We want it at just the filename (train_000000.npy)
                files_to_download.append((s3_key, filename))

    if not files_to_download:
        print(f"\nAll files for shards {sorted(shard_indices)} already exist locally.")
        print(f"Skipped {len(skipped_files)} files.")
        return

    print(
        f"\nDownloading {len(files_to_download)} files from shards: {sorted(shard_indices)}"
    )
    if skipped_files:
        print(f"Skipping {len(skipped_files)} existing files")
    print(f"Bucket: {bucket_name}")
    print("-" * 40)

    try:
        # Download files concurrently
        tasks = []
        for s3_key, local_path in files_to_download:
            print(f"Queuing download: {s3_key} -> {Path(local_path).name}")

            # Create parent directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # Create download task
            # s3_get_object will download using s3_key and save to that path
            # After download completes, we'll move it to local_path
            task = comms.s3_get_object(
                key=s3_key,
                bucket=bucket,
                load_data=False,  # Save to disk, don't load into memory
                show_progress=True,
                timeout=600,
            )
            tasks.append((task, s3_key, local_path))

        print(f"\nStarting concurrent downloads...")
        print("-" * 40)

        # Run all downloads concurrently
        success_count = 0
        fail_count = 0
        for task, s3_key, local_path in tasks:
            try:
                result = await task
                if result:
                    # When load_data=False, s3_get_object returns the path where it saved the file
                    downloaded_path = Path(result)
                    target_path = Path(local_path)

                    # Check if file ended up at the target location already
                    if target_path.exists():
                        file_size = target_path.stat().st_size
                        print(f"✓ Downloaded {target_path} ({file_size:,} bytes)")
                        success_count += 1
                    elif downloaded_path.exists():
                        # File is at a different location, move it
                        if downloaded_path != target_path:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(downloaded_path), str(target_path))
                            # Clean up empty directories
                            try:
                                downloaded_path.parent.rmdir()
                            except:
                                pass

                        if target_path.exists():
                            file_size = target_path.stat().st_size
                            print(f"✓ Downloaded {target_path} ({file_size:,} bytes)")
                            success_count += 1
                        else:
                            print(f"✗ Failed to move {downloaded_path} to {local_path}")
                            fail_count += 1
                    else:
                        print(f"✗ File not found after download: {downloaded_path}")
                        fail_count += 1
                else:
                    print(f"✗ Failed to download {s3_key}")
                    fail_count += 1
            except Exception as e:
                print(f"✗ Error downloading {s3_key}: {e}")
                fail_count += 1

        print("-" * 40)
        print(f"Download complete: {success_count} successful, {fail_count} failed")

        if success_count > 0:
            print(f"\nFiles saved to: {output_path.absolute()}")

    finally:
        # Restore original directory
        try:
            os.chdir(original_dir)
        except:
            pass

        await comms.close_all_s3_clients()

        # Clean up temp directory if it exists
        if hasattr(comms, "temp_dir") and os.path.exists(comms.temp_dir):
            try:
                shutil.rmtree(comms.temp_dir)
                print(f"Cleaned up temporary directory: {comms.temp_dir}")
            except Exception as e:
                print(
                    f"Warning: Could not clean up temp directory {comms.temp_dir}: {e}"
                )


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Parsed arguments with shard_indices and output_path
    """
    parser = argparse.ArgumentParser(
        description="Download dataset shards from R2 storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                          # Downloads shards 0 and 1 to default path
    %(prog)s 0 1 2 3                  # Downloads shards 0, 1, 2, and 3
    %(prog)s --path /custom/path 0 1  # Downloads to custom path
    %(prog)s -p ./data 2 3 4          # Downloads to ./data

Environment variables:
    DATASET_BINS_PATH    Default output directory (if not specified via --path)
    R2_DATASET_*         R2 credentials (required)
        """,
    )

    parser.add_argument(
        "shard_indices",
        nargs="*",
        type=int,
        help="Shard indices to download (default: 0 1)",
    )

    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Output directory path (default: from DATASET_BINS_PATH or 'remote/tokenized/')",
    )

    args = parser.parse_args()

    # Default shard indices if none provided
    if not args.shard_indices:
        args.shard_indices = [0, 1]

    # Remove duplicates and sort
    args.shard_indices = sorted(set(args.shard_indices))

    # Filter out negative indices
    valid_indices = [idx for idx in args.shard_indices if idx >= 0]
    if len(valid_indices) < len(args.shard_indices):
        invalid = [idx for idx in args.shard_indices if idx < 0]
        print(f"Warning: Ignoring negative shard indices: {invalid}")
    args.shard_indices = valid_indices

    if not args.shard_indices:
        print("Error: No valid shard indices provided")
        sys.exit(1)

    # Determine output path - this should match DATASET_BINS_PATH
    if args.path is None:
        # Try environment variable first, default to "remote/tokenized/"
        args.path = os.getenv("DATASET_BINS_PATH", "remote/tokenized/")

    # Remove any trailing slashes for consistency
    args.path = args.path.rstrip("/")

    return args


if __name__ == "__main__":
    print("Dataset Shard Downloader")
    print("=" * 40)

    # Check for required environment variables
    required_vars = [
        "R2_DATASET_ACCOUNT_ID",
        "R2_DATASET_BUCKET_NAME",
        "R2_DATASET_READ_ACCESS_KEY_ID",
        "R2_DATASET_READ_SECRET_ACCESS_KEY",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these in your .env file or export them")
        sys.exit(1)

    # Parse arguments
    args = parse_arguments()

    print(f"Shards to download: {args.shard_indices}")
    print(f"Output path: {args.path}/")

    # Run the async download function
    asyncio.run(download_files(args.shard_indices, args.path))
    print("\nAll downloads completed!")
