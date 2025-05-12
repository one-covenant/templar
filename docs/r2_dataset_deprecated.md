## System Requirements
> You will need to make sure you have adequate bandwidth. 17.51 TB of data (as of 2/6/2025) will be transferred through your machine but NOT be stored locally. The higher the internet speed and more workers/cores you can throw into the process, the faster the dataset will be finished. Consider that this process will take 12 or more hours, so use screen/tmux accordingly. Re-run the transfer command for the process to verify the files and pick up where it left off.

Recommend workhorse:
Network: 1gbps+
Local Storage: 100gb
RAM: 4 GB+ (this process is not memory intensive)
Cores: 8+
Estimated Download Time: 12–18 hours

Instructions
r2 bucket dataset population

```bash
#!/bin/bash

# Configure dataset bucket name to use.
export DATABUCKET="dataset"

# clone repo
git clone https://github.com/distributedstatemachine/HuggingFaceModelDownloader
cd HuggingFaceModelDownloader

# create local .env file for R2 account creds
tee .env << 'EOF'
R2_ACCOUNT_ID=
R2_WRITE_ACCESS_KEY_ID=
R2_WRITE_SECRET_ACCESS_KEY=
EOF

# install go
wget https://go.dev/dl/go1.23.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.23.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# check go version
go version

# gather CPU count to use for transfer
export CPUCOUNT=$(grep -c '^processor' /proc/cpuinfo)

export DATABUCKET="dataset"
# Configure dataset bucket name to use.

# start transfer
go run main.go -d "HuggingFaceFW/fineweb-edu-score-2" --r2 --skip-local -c $CPUCOUNT  --branch v1.2.0 --r2-bucket $DATABUCKET

# check corrupted files
go run main.go -d "HuggingFaceFW/fineweb-edu-score-2" --r2 --cleanup-corrupted --branch v1.2.0 --r2-bucket $DATABUCKET

# if needed, re-transfer
go run main.go -d "HuggingFaceFW/fineweb-edu-score-2" --r2 --skip-local -c $CPUCOUNT --r2-bucket $DATABUCKET
final config for shard and metadata files
#!/bin/bash
# Return to the templar repo
cd ~/templar

# modify local _shard_sizes.json using the $DATABUCKET we configured previously
sed -i 's|80f15715bb0b882c9e967c13e677ed7d/|{$DATABUCKET}/|g' _shard_sizes.json

# Finally clear any local cache from previous runs and prompt miner to request new data from the r2 dataset bucket on next run
rm ./.cache/tplr/*
You are now ready to use the dataset for your miner and set your own read only API keys for accessing the dataset bucket

#!/bin/bash

export R2_DATASET_ACCOUNT_ID=$R2_ACCOUNT_ID
export R2_DATASET_BUCKET_NAME=$DATABUCKET
export R2_DATASET_READ_ACCESS_KEY_ID=
export R2_DATASET_READ_SECRET_ACCESS_KEY=
Reference images
r2 Bucket view image

dataset bucket top level view image

dataset folder image
```