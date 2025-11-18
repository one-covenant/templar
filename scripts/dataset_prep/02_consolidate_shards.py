import argparse
import asyncio
import hashlib
import io
import os
import struct
import time

import boto3
import numpy as np
from tqdm.auto import tqdm

import tplr
from neurons import miner
from tplr import comms, hparams, sharded_dataset


def tokens_handler(x: np.ndarray) -> int:
    """
    Takes the incoming numpy array and creates the sample_id
    corresponding to that array

    Args:
        x: A numpy array representing a data slice where
           len == seq_len

    Returns:
        The hashed representation of this slice
    """
    x = x.tobytes()
    x = hashlib.blake2b(x, digest_size=8).digest()
    x = struct.unpack("<Q", x)[0]
    return x


async def run_preprocessing(
    args, seq_len: int = 2048, token_dtype: np.dtype = np.uint16
) -> None:
    """
    Gathers .npy shards and creates 'sample_ids.bin' to represent the hash

    Since this is now a Templar internal responsibility, no sha256 checking
    """

    config = miner.Miner.miner_config()
    comms_ = comms.Comms(
        wallet=None,
        config=config,
        neuid=config.netuid,
        hparams=hparams.load_hparams(),
    )
    bucket = comms_.get_own_bucket("dataset", "read")
    tokens_file, ids_file = sharded_dataset.SharedShardedDataset.locate_shards(0)
    if not os.path.exists(tokens_file):
        try:
            print("Downloading first shard")

            _ = await asyncio.create_task(
                comms_.s3_get_object(
                    tokens_file,
                    bucket,
                    load_data=False,
                )
            )
            tplr.logging.info("Shard downloaded")
        except Exception as e:
            tplr.logger.error(e)

    args.r2_endpoint_url = f"https://{args.r2_endpoint_url}.r2.cloudflarestorage.com"
    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        endpoint_url=args.r2_endpoint_url,
        aws_access_key_id=args.r2_access_key_id,
        aws_secret_access_key=args.r2_secret_access_key,
        region_name="auto",
    )

    # Check existing shards in R2
    try:
        response = s3_client.list_objects_v2(
            Bucket=args.r2_bucket, Prefix=os.path.join(args.r2_prefix, "train_")
        )
        contents = response.get("Contents", [])
        shards = list(filter(lambda c: c["Key"].endswith(".npy"), contents))
        shards = len(shards)
        print(f"Shards found: {shards}")
    except Exception as e:
        print(f"Could not list objects in R2 bucket: {e}")
        shards = 0

    file_getter = asyncio.create_task(asyncio.sleep(0))
    for i in range(shards):
        await file_getter

        t1 = time.perf_counter()

        new_tokens_file, new_ids_file = None, None
        if i + 1 < shards:
            new_tokens_file, new_ids_file = (
                sharded_dataset.SharedShardedDataset.locate_shards(i + 1)
            )
            file_getter = asyncio.create_task(
                comms_.s3_get_object(
                    new_tokens_file,
                    bucket,
                    load_data=False,
                )
            )

        # Load tokens - if .npy file, use np.load to respect embedded dtype
        if tokens_file.endswith(".npy"):
            tokens_view = np.load(tokens_file, mmap_mode="r", allow_pickle=False)
            # Ensure it's uint32 (step 01 saves as uint32)
            if tokens_view.dtype != np.uint32:
                tqdm.write(
                    f"Warning: Shard {i} has dtype {tokens_view.dtype}, expected uint32. "
                    f"Converting (this may indicate a preprocessing mismatch)."
                )
                tok_u32 = tokens_view.astype(np.uint32)
            else:
                tok_u32 = tokens_view
        else:
            # Raw binary file - use specified dtype then reinterpret as uint32
            tokens_view = np.memmap(tokens_file, dtype=token_dtype, mode="r")
            tok_u32 = tokens_view.view(np.uint32)

        # Only create sample IDs for complete sequences (discard partial sequence at end)
        total_tokens = tok_u32.shape[0]
        num_complete_samples = total_tokens // seq_len

        # Warn if there's a partial sequence that will be discarded
        remainder = total_tokens % seq_len
        if remainder > 0:
            tqdm.write(
                f"Warning: Shard {i} has {remainder} tokens remaining after chunking "
                f"(total: {total_tokens}, seq_len: {seq_len}). "
                f"Creating {num_complete_samples} complete samples."
            )

        raw_idx = np.arange(0, num_complete_samples * seq_len + 1, seq_len)
        starts = raw_idx[:-1]
        ends = raw_idx[1:]

        bits = [
            tokens_handler(tok_u32[start:end])
            for start, end in tqdm(zip(starts, ends), total=len(starts))
        ]
        sample_ids = np.stack(bits).view(np.uint64)

        buffer = io.BytesIO()
        np.save(buffer, sample_ids)
        buffer.seek(0)

        filename = ids_file.split("/")[-1]
        key = os.path.join(args.r2_prefix, filename)
        s3_client.upload_fileobj(buffer, args.r2_bucket, key)
        tqdm.write(f"Uploaded sample_ids {i} to R2: s3://{args.r2_bucket}/{key} ")

        del tokens_view, sample_ids
        os.remove(tokens_file)
        tokens_file = new_tokens_file
        ids_file = new_ids_file

    print(f"sample_ids.bin written in {time.perf_counter() - t1:.1f}s")

    return


async def main() -> None:
    """
    Run the process that hashes the sample_ids

    Raises:
        ValueError: If sequence_length is invalid
    """

    parser = argparse.ArgumentParser(
        description="Create sample_ids.bin for each .npy shard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Model context length for sequence chunking",
    )

    parser.add_argument(
        "--token_dtype",
        type=str,
        default="uint16",
        choices=["uint8", "uint16", "uint32"],
        help="Data type used in the shards",
    )

    # R2 arguments
    parser.add_argument(
        "--r2_bucket",
        default=os.getenv("R2_DATASET_BUCKET_NAME"),
        help="R2 bucket name",
    )
    parser.add_argument(
        "--r2_prefix",
        default="tokenized/",
        help="R2 prefix for shards",
    )
    parser.add_argument(
        "--r2_endpoint_url",
        default=os.getenv("R2_DATASET_ACCOUNT_ID"),
        help="R2 endpoint URL",
    )
    parser.add_argument(
        "--r2_access_key_id",
        default=os.getenv("R2_DATASET_WRITE_ACCESS_KEY_ID"),
        help="R2 access key ID",
    )
    parser.add_argument(
        "--r2_secret_access_key",
        default=os.getenv("R2_DATASET_WRITE_SECRET_ACCESS_KEY"),
        help="R2 secret access key",
    )

    args = parser.parse_args()

    # Convert string dtype to numpy dtype
    token_dtype = getattr(np, args.token_dtype)

    # Validate inputs
    if args.seq_len <= 0:
        raise ValueError("Sequence length must be positive")

    print("Configuration:")
    print(f"  • Shards path: {args.r2_prefix}")
    print(f"  • Sequence length: {args.seq_len}")
    print(f"  • Token dtype: {args.token_dtype}")
    print()

    success = await run_preprocessing(args, args.seq_len, token_dtype)

    if not success:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
