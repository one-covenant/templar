# Shared Sharded Dataset (2 trillion tokens)

08/02/2025

This guide explains how to access the shards for the 2T tokens dataset.

As we scale up to larger models, we need more data for the hungry model to consume. For the next run we'll have 6x more data, with about 2TB of storage required for everything on disk. At the largest model sizes, we'll need even more than that!

With our miners and validators in mind, we wanted to avoid disk storage pressure but maintain the advantages of having the data locally. To do that, we will be blocking the download of the first ~430GB shard (S1). We're seeing this should take about 10-20m under the current codebase defaults, assuming 32+ cpus. The next shard will download in the background (S2).

At the end of the allocated windows, the `SharedShardedDatasetManager` will swap the datasets and the dataloader will be refreshed. The upcoming shard (S3) will be downloaded automatically in the background, and the cycle repeats through the end of the run (S4, S5, ...).

To keep disk utilisation minimal, shards are deleted after their respective window is over. If the model does another pass over the data, the initial shard is redownloaded and the process continues. To avoid shard deletion on 4TB+ local storage disks, see the final section on [Self-Hosting](#self-hosting).

## Dataset Information

The Shared Sharded dataset is based on the [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) dataset on Hugging Face, part of the research available at [DataComp](https://data.commoncrawl.org/contrib/datacomp/index.html).

For the fastest training, our optimized version includes:

- Pretokenized numpy arrays in .npy files
- Sample ID arrays provided via .npy files

## System Requirements

The dataset transfer requires adequate bandwidth. Files are temporarily processed on your machine but not permanently stored locally. Higher internet speeds and more worker threads will significantly accelerate the process.

Expect to simultaneously hold 2+ dataset shards at one time. Each shard is 430GB. As one shard downloads, another is downloading. When a swap happens, both will be on disk and another one may be downloading to the tmp directory.

### Recommended Hardware

- **Network**: 1gbps+
- **Local Storage**: 750GB (for temporary processing)
- **RAM**: 4GB+
- **CPU Cores**: 100+ (more cores = faster downloading the datasets)
- **Estimated Download Time**: 20 minutes per shard (depending on network speed and cpus)

## Setup Instructions

### Modify your Cloudflare R2 `dataset` Bucket settings

Append the following env keys:

```bash
R2_DATASET_ACCOUNT_ID=8af7f92a8a0661cf7f1ac0420c932980
R2_DATASET_BUCKET_NAME=mixed-dataset-migration
R2_DATASET_READ_ACCESS_KEY_ID=e70cd26850f697479bbb5fd9413713f4
R2_DATASET_READ_SECRET_ACCESS_KEY=11e3364d6ef70e44d671863fb6de32d474aa6220fa2c9c3df45c5e012ebfbda3
DATASET_BINS_PATH="anneal/"
```

That's all! For the minimum setup, the dataset has been built and is available by providing those keys. The `SharedShardedDatasetManager` takes care of everything else. Unlike previously where user steps were required, now you only have to point to a bucket with the shards in it.

*Important*:
You must use the [Dataset Migration](#dataset-migration) tool. This copies the files to your bucket to avoid any potential rate-limiting that could arise from the shared bucket. In that scenario, be sure to update `R2_DATASET_ACCOUNT_ID`, `R2_DATASET_READ_ACCESS_KEY_ID`, and `R2_DATASET_READ_SECRET_ACCESS_KEY`.

## Troubleshooting

### Common Issues

1. **Network Interruptions**: The comms module, called in the `SharedShardedDatasetManager`, will automatically retry
2. **Slow Downloads**: Consider:

   - Edit `comms.py` in the `download_large_file` function and hard-code the `max_workers=total_chunks`
   - Using a machine with higher network bandwidth
   - Using a machine with more CPUs
3. **Space Issues**: If you run out of space during temporary file processing, ensure you have at least 225GB of free space or use a machine with more storage.
4. **HTTP Timeout Errors**: These are handled automatically with retries. If you see persistent timeouts, there might be network connectivity issues to either Hugging Face or Cloudflare.

## Performance Considerations

- **Optimizing Download Speed**: If you were to decrease the `chunk_size` to something like 2 * 1024**3, you could use 100 workers in parallel
- **Memory Usage**: Not sure but use plenty of RAM if you modify anything
- **Storage I/O**: Using SSD storage for temporary files will significantly improve performance

## Self-Hosting `<a name="self-hosting"></a>`

There are two useful directives when self hosting:

- Self bucket management with the CloudFlare migration tool
- Preventing `os.remove` during the dataset swap

### R2 Self-Hosting `<a name="dataset-migration"></a>`

For self hosting, you have two options: using the CloudFlare migration tool (recommended) or using rclone for manual migration.

#### Option 1: CloudFlare Migration Tool (Recommended)

Use the CloudFlare migration tool for the easiest setup. Here are the key-value pairs for the UI:

#### Page 1

- Bucket Information
  `Source bucket provider`: `S3-Compatible Storage`
  `Bucket name`: `mixed-dataset-migration`
  `S3-compatible endpoint URL`: `https://8af7f92a8a0661cf7f1ac0420c932980.r2.cloudflarestorage.com/mixed-dataset-migration`
- Required Credentials
  `Access Key ID`: `e70cd26850f697479bbb5fd9413713f4`
  `Secret Access Key`: `11e3364d6ef70e44d671863fb6de32d474aa6220fa2c9c3df45c5e012ebfbda3`

#### Page 2

- Select destination R2 bucket
  `Bucket name`: `mixed-dataset-migration`
  `Access Key ID`: your_write_id
  `Access Key`: your_secret_write_id
  `Overwrite files?`: `Yes, overwrite (recommended)`

#### Option 2: Manual Migration with rclone

If you encounter issues with the CloudFlare migration tool or need more control over the migration process, you can use rclone. This method also allows you to copy specific shards for testing purposes.

**Note**: Migration speed depends heavily on your internet connection. A faster connection (10Gbps+) will significantly reduce transfer times.

##### Install rclone
```bash
curl https://rclone.org/install.sh | sudo bash
```

##### Configure rclone for source and destination buckets
```bash
# Configure source (read-only)
rclone config create r2-source s3 \
  provider=Cloudflare \
  access_key_id=e70cd26850f697479bbb5fd9413713f4 \
  secret_access_key=11e3364d6ef70e44d671863fb6de32d474aa6220fa2c9c3df45c5e012ebfbda3 \
  endpoint=https://8af7f92a8a0661cf7f1ac0420c932980.r2.cloudflarestorage.com \
  acl=private

# Configure destination (your bucket)
rclone config create r2-dest s3 \
  provider=Cloudflare \
  access_key_id=<your-write-access-key-id> \
  secret_access_key=<your-write-secret-access-key> \
  endpoint=https://<your-account-id>.r2.cloudflarestorage.com \
  acl=private
```

##### Copy all shards (Full Migration)
```bash
# Copy entire anneal directory (all shards and sample IDs)
rclone copy r2-source:mixed-dataset-migration/anneal/ r2-dest:<your-bucket-name>/anneal/ \
  --transfers 32 \
  --checkers 16 \
  --progress
```

##### Copy specific shards (Partial Migration for Testing)
If you want to test with the current anneal shard:
```bash
# Copy anneal shard 4 and its sample IDs
rclone copy r2-source:mixed-dataset-migration/anneal/anneal_000004.npy r2-dest:<your-bucket-name>/anneal/ --progress
rclone copy r2-source:mixed-dataset-migration/anneal/sample_ids_anneal_000004.npy r2-dest:<your-bucket-name>/anneal/ --progress
```

After migration, update your environment variables to point to your bucket:
```bash
R2_DATASET_ACCOUNT_ID=<your-account-id>
R2_DATASET_BUCKET_NAME=<your-bucket-name>
R2_DATASET_READ_ACCESS_KEY_ID=<your-read-access-key-id>
R2_DATASET_READ_SECRET_ACCESS_KEY=<your-read-secret-access-key>
```

### Keeping inactive shards on disk

The `SharedShardedDatasetManager` class does handle object deletion for inactive shards.

To prevent this:

* Navigate to `src/tplr/sharded_dataset`
* Comment out the loop starting with `if old_dataset and self.rank == 0:` with `#` at the line beginning

Since that loop (with `os.remove`) would have deleted the inactive shards, now they will stay on disk.

The total disk utilization of all shards is > 4TB.
