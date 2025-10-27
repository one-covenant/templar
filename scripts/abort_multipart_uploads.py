#!/usr/bin/env python3
# Script to abort ongoing multipart uploads in R2 bucket

import asyncio
import os
import sys
from pathlib import Path

from aiobotocore.session import get_session
from dotenv import load_dotenv

# Find and load the correct .env file
env_path = Path(__file__).parent / "../.env"
if not env_path.exists():
    raise FileNotFoundError(f"Required .env file not found at {env_path}")

# Load environment variables
load_dotenv(env_path, override=True)

# Add parent directory to path to import tplr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tplr


async def abort_multipart_uploads():
    """Abort all ongoing multipart uploads in the R2 bucket"""

    # Validate required environment variables
    required_vars = [
        "R2_GRADIENTS_ACCOUNT_ID",
        "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
        "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
    ]

    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Get credentials from environment
    account_id = os.environ["R2_GRADIENTS_ACCOUNT_ID"]
    access_key_id = os.environ["R2_GRADIENTS_WRITE_ACCESS_KEY_ID"]
    secret_access_key = os.environ["R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY"]

    # Initialize S3 client
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        region_name="enam",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=tplr.config.client_config,
    ) as client:
        print("Listing multipart uploads...")

        try:
            # List all multipart uploads
            response = await client.list_multipart_uploads(Bucket=account_id)

            uploads = response.get("Uploads", [])

            if not uploads:
                print("No ongoing multipart uploads found")
                return

            print(f"Found {len(uploads)} ongoing multipart uploads")

            # Abort each upload
            aborted_count = 0
            for upload in uploads:
                key = upload["Key"]
                upload_id = upload["UploadId"]

                try:
                    await client.abort_multipart_upload(
                        Bucket=account_id, Key=key, UploadId=upload_id
                    )
                    print(f"✓ Aborted: {key}")
                    aborted_count += 1
                except Exception as e:
                    print(f"✗ Failed to abort {key}: {e}")

            print(
                f"\nSuccessfully aborted {aborted_count}/{len(uploads)} multipart uploads"
            )

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    print("Starting abort of multipart uploads...")
    asyncio.run(abort_multipart_uploads())
