# Miner Setup

> **⚠️ 100% Burn Period Active**: Miners are not being evaluated and no emissions are being distributed. The Covenant72B training phase is complete. **Templar: Crusades** - a new competition system - is coming soon. See the main [README](../README.md) for current status.

This document provides a comprehensive guide on how to set up and run a miner using `miner.py`. Miners are crucial components of **τemplar**, responsible for training the model on assigned data subsets and sharing their gradients with peers.

## Table of Contents

- [Miner Setup](#miner-setup)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Running the Miner](#running-the-miner)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Hardware Requirements](#hardware-requirements)
    - [Network Options](#network-options)
    - [InfluxDB Configuration](#influxdb-configuration)
  - [Monitoring](#monitoring)
    - [Logs](#logs)
    - [Performance](#performance)
  - [Troubleshooting](#troubleshooting)
  - [Miner Operations](#miner-operations)
    - [Model Synchronization](#model-synchronization)
    - [Training Process](#training-process)
    - [Gradient Sharing](#gradient-sharing)

---

## Introduction

This guide will help you set up and run a miner for **τemplar**. We'll cover both the recommended Docker Compose method and manual installation for environments where Docker is not preferred.

---

## Prerequisites

- **NVIDIA GPU** with CUDA support
  - **Minimum required: 8x B200 GPUs**
- **Ubuntu** (or Ubuntu-based Linux distribution)
- **Git**
- **Hugging Face Authentication**:
  - Create a Hugging Face account and generate a token at https://huggingface.co/settings/tokens
  - Accept the Gemma model terms at https://huggingface.co/google/gemma-3-270m (required for tokenizer access)
  - Set `HF_TOKEN` environment variable with your token
- **Cloudflare R2 Bucket Configuration**:
  - **Dataset Setup**: Please refer to [Shared Sharded Dataset Documentation](./shared_sharded_dataset.md) for complete dataset setup instructions, including:
    - R2 bucket settings
    - Dataset download process
    - No pre-download is required, but bucket syncing is optional and recommended
  - **Gradient Bucket Setup**:
    1. **Create a Bucket**: Name it the same as your **account ID** and set the **region** to **ENAM**.
    2. **Generate Tokens**:
       - **Read Token**: Admin Read permissions.
       - **Write Token**: Admin Read & Write permissions.
    3. **Store Credentials**: You'll need these for the `.env` file.

---

## Running the Miner

> Note: Using Ansible (Automated Setup)
>
> For automated deployment across multiple hosts or multi-GPU configurations, you can use our Ansible playbook. This method is particularly useful for:
>
> - Deploying to multiple servers
> - Managing multi-GPU setups
> - Automating the entire setup process
>
> See the [Ansible Setup Guide](./miner-setup-ansible.md) for detailed instructions.

### Instructions

1. **Install System Dependencies**:

   ```bash
   # Add Python 3.11 repository
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update

   # Install required packages
   sudo apt-get install python3.11 python3.11-venv git
   ```

  *PM2 Support Installation

  ```bash
   # Install required packages
   apt update && apt upgrade -y && apt-get install -y nano git python3-pip jq npm && npm install pm2 -g && pm2 update
   ```

2. **Install NVIDIA CUDA Drivers**:

   Install the appropriate NVIDIA CUDA drivers for your GPU.

3. **Clone the Repository**:

   ```bash
   git clone https://github.com/one-covenant/templar.git
   cd templar
   ```

4. **Set Up Python Environment**:

   ```bash
   # Create virtual environment
   python3.11 -m venv .venv
   source .venv/bin/activate

   # Upgrade pip
   pip install --upgrade pip

   # Install PyTorch with CUDA support
   pip install torch --index-url https://download.pytorch.org/whl/cu118


   # Install uv tool (if needed)
   pip install uv
   ```

  *PM2 Support Installation

  ```bash
   # Install uv and configure venv
   pip install uv && uv python install 3.11 && uv python pin 3.11 && uv venv .venv
   source .venv/bin/activate

   # Install PyTorch with CUDA support
   uv pip install torch --index-url https://download.pytorch.org/whl/cu118\

   # uv sync to install required packages
   uv sync --extra all
   ```

5. **Create and Register Wallets**:

   ```bash
   # Create coldkey
   btcli wallet new_coldkey --wallet.name default --n-words 12

   # Create and register hotkey
   btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner --n-words 12
   btcli subnet register --wallet.name default --wallet.hotkey miner --netuid <netuid> --subtensor.network <network>
   ```

6. **Log into Weights & Biases (WandB)**:

   ```bash
   wandb login your_wandb_api_key
   ```

7. **Set Environment Variables**:

   Export necessary environment variables or create a `.env` file in the project root.

   ```bash
   export HF_TOKEN=your_huggingface_token  # Required for tokenizer access
   export WANDB_API_KEY=your_wandb_api_key
   export INFLUXDB_TOKEN=your_influxdb_token
   export NODE_TYPE=your_node_type
   export WALLET_NAME=your_wallet_name
   export WALLET_HOTKEY=your_wallet_hotkey
   # GPU is automatically assigned by Docker (GPUs 0,1,2 for miner)
   export NETWORK=your_network
   export NETUID=your_netuid
   export DEBUG=your_debug_setting
   
   # Gradients R2 credentials
   export R2_GRADIENTS_ACCOUNT_ID=your_r2_account_id
   export R2_GRADIENTS_BUCKET_NAME=your_r2_bucket_name
   export R2_GRADIENTS_READ_ACCESS_KEY_ID=your_r2_read_access_key_id 
   export R2_GRADIENTS_READ_SECRET_ACCESS_KEY=your_r2_read_secret_access_key
   export R2_GRADIENTS_WRITE_ACCESS_KEY_ID=your_r2_write_access_key_id
   export R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY=your_r2_write_secret_access_key

   # Dataset R2 credentials - You may set up your own Shared Sharded Dataset, but must at minimum set these keys
   # See docs/shared_sharded_dataset.md for instructions
   export R2_DATASET_ACCOUNT_ID="8af7f92a8a0661cf7f1ac0420c932980"
   export R2_DATASET_BUCKET_NAME="mixed-dataset-migration"
   export R2_DATASET_READ_ACCESS_KEY_ID="e70cd26850f697479bbb5fd9413713f4"
   export R2_DATASET_READ_SECRET_ACCESS_KEY="11e3364d6ef70e44d671863fb6de32d474aa6220fa2c9c3df45c5e012ebfbda3"
   export DATASET_BINS_PATH="anneal/"


   # Aggregator R2 credentials
   export R2_AGGREGATOR_ACCOUNT_ID="8af7f92a8a0661cf7f1ac0420c932980"
   export R2_AGGREGATOR_BUCKET_NAME="aggregator"
   export R2_AGGREGATOR_READ_ACCESS_KEY_ID="bb4b9f02a64dacead181786b8f353b67"
   export R2_AGGREGATOR_READ_SECRET_ACCESS_KEY="f50761d0fbb0773c55f61debdf87439735c32c096fe4b1ab6aa6bfb7f52aa30b"
   
   export GITHUB_USER=your_github_username
   ```

8. **Run the Miner**:

   ```bash
   torchrun --standalone --nnodes 1 --nproc_per_node 8 \
     neurons/miner.py \
     --wallet.name <wallet_name> \
     --wallet.hotkey <hotkey> \
     --device cuda \
     --netuid 3 \
     --subtensor.network <network> \
     --use_wandb
   ```

  *PM2 Support Installation

  ```bash
   pm2 start "torchrun --standalone --nnodes 1 --nproc_per_node 8 neurons/miner.py \
   --wallet.name <wallet_name> \
   --wallet.hotkey <hotkey> \
   --device cuda \
   --subtensor.network <network> \
   --netuid 3 \
   --use_wandb" \
   --name sn3miner
  ```

---

## Configuration

### Environment Variables

When using Docker Compose, set the following variables in the `docker/.env` file:

```dotenv:docker/.env
# Required: Hugging Face token for tokenizer access
HF_TOKEN=your_huggingface_token

# Add your Weights & Biases API key
WANDB_API_KEY=your_wandb_api_key
INFLUXDB_TOKEN=your_influxdb_token

# Cloudflare R2 Credentials
R2_ACCOUNT_ID=your_r2_account_id

R2_READ_ACCESS_KEY_ID=your_r2_read_access_key_id
R2_READ_SECRET_ACCESS_KEY=your_r2_read_secret_access_key

R2_WRITE_ACCESS_KEY_ID=your_r2_write_access_key_id
R2_WRITE_SECRET_ACCESS_KEY=your_r2_write_secret_access_key

# Wallet Configuration
WALLET_NAME=default
WALLET_HOTKEY=your_miner_hotkey_name

# Network Configuration
NETWORK=finney
NETUID=3

# GPU Configuration (automatically handled by Docker)
# Miner service uses GPUs 0, 1, and 2 from the host

# Additional Settings
DEBUG=false
```

**Note**: The R2 permissions remain unchanged from previous configurations.

### Hardware Requirements

- **GPU Requirements**:
  - **Minimum required: 8x B200 GPUs** (as defined in min_compute.yml)
  - **Minimum CPU**: 64 cores, 3.5 GHz (recommended: 128 cores, 4.0 GHz)
  - **Minimum RAM**: 1200 GB (recommended: 1500 GB)
  - **Minimum Network**: 1024 Mbps download/upload bandwidth
- **Storage**: 1000GB minimum, 2000GB recommended for model and data
- **Network**: Stable internet connection with good bandwidth

### Network Options

- **Mainnet (Finney)**:
  - Network: `finney`
  - Netuid: `3`
- **Testnet**:
  - Network: `test`
  - Netuid: `223`
- **Local**:
  - Network: `local`
  - Netuid: `1`

### InfluxDB Configuration

Optional InfluxDB configuration variables include:

- `INFLUXDB_TOKEN`: Authentication token
- `INFLUXDB_HOST`: Custom host address
- `INFLUXDB_PORT`: Connection port (default 8086)
- `INFLUXDB_DATABASE`: Database name
- `INFLUXDB_ORG`: Organization identifier

Example configuration:

```bash
INFLUXDB_HOST=custom-influxdb-host.example.com
INFLUXDB_PORT=8086
INFLUXDB_DATABASE=custom-database
INFLUXDB_ORG=custom-org
INFLUXDB_TOKEN=your-influxdb-token
```

These settings are optional and will fall back to default values if not provided.

---

## Monitoring

### Logs

- **Docker Logs**:

  ```bash
  docker logs -f templar-miner-${WALLET_HOTKEY}
  ```

- **Weights & Biases**:

  - Ensure `--use_wandb` is enabled
  - Monitor training metrics and performance on your WandB dashboard

### Performance

Keep an eye on:

- GPU utilization
- Memory usage
- Network bandwidth
- Training progress
- Rewards and weights

---

## Troubleshooting

- **CUDA Out of Memory**: Reduce `--actual_batch_size` in your run command.
- **Network Synchronization Issues**: Verify your network connection and ensure the correct `NETWORK` and `NETUID` are set.
- **Registration Failures**: Make sure your wallet is properly registered and funded.

---

## Miner Operations

### Model Synchronization

- The miner synchronizes its model with the latest global state at startup.
- Attempts to load the latest checkpoint from the validator with the highest stake.

### Training Process

- Data is deterministically assigned based on the miner's UID and the current window.
- The miner trains on its assigned data and computes gradients.

### Gradient Sharing

- Gradients are compressed and shared with peers via the communication module.
- The miner gathers gradients from peers, decompresses them, and updates its model.

---
