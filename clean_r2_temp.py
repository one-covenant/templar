import os

import boto3
from dotenv import load_dotenv

load_dotenv('.env')

# Clean gradients bucket
bucket_configs = [
    {
        'name': 'gradients',
        'bucket': os.getenv('R2_GRADIENTS_BUCKET_NAME'),
        'account_id': os.getenv('R2_GRADIENTS_ACCOUNT_ID'),
        'access_key': os.getenv('R2_GRADIENTS_WRITE_ACCESS_KEY_ID'),
        'secret_key': os.getenv('R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY'),
    }
]

for config in bucket_configs:
    bucket_name = config['bucket']
    account_id = config['account_id']
    access_key = config['access_key']
    secret_key = config['secret_key']
    name = config['name']
    
    if not all([bucket_name, account_id, access_key, secret_key]):
        print(f"Missing credentials for {name} bucket")
        continue
    
    endpoint = f'https://{account_id}.r2.cloudflarestorage.com'
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='auto'
    )
    
    try:
        # Delete all objects
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        deleted = 0
        for page in pages:
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects:
                    s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})
                    deleted += len(objects)
        
        # Abort incomplete multipart uploads
        response = s3.list_multipart_uploads(Bucket=bucket_name)
        if 'Uploads' in response:
            for upload in response['Uploads']:
                s3.abort_multipart_upload(
                    Bucket=bucket_name,
                    Key=upload['Key'],
                    UploadId=upload['UploadId']
                )
            print(f"{name}: Deleted {deleted} objects, aborted {len(response['Uploads'])} multipart uploads")
        else:
            print(f"{name}: Deleted {deleted} objects")
    except Exception as e:
        print(f"Error cleaning {name}: {e}")


