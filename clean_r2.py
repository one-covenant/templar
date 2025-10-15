#!/usr/bin/env python3
import os

from dotenv import load_dotenv

load_dotenv()

import boto3

# Use the correct environment variable names
account_id = os.getenv('R2_GRADIENTS_ACCOUNT_ID')
access_key = os.getenv('R2_GRADIENTS_WRITE_ACCESS_KEY_ID')  # Use WRITE key for deletion
secret_key = os.getenv('R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY')
bucket_name = os.getenv('R2_GRADIENTS_BUCKET_NAME')

print(f"Account ID: {account_id}")
print(f"Bucket: {bucket_name}")

if not all([account_id, access_key, secret_key, bucket_name]):
    print("Missing R2 credentials")
    exit(1)

s3 = boto3.client(
    's3',
    endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='auto'
)

try:
    deleted_count = 0
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            objects = [{'Key': obj['Key']} for obj in page['Contents']]
            if objects:
                s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})
                deleted_count += len(objects)
                print(f"Deleted batch of {len(objects)} objects...")
    
    print(f"✅ Total deleted: {deleted_count} objects from {bucket_name}")
except Exception as e:
    print(f"Error: {e}")




