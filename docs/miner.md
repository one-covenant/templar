# Miner Setup

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
  - Minimum H100 recommended
- **Ubuntu** (or Ubuntu-based Linux distribution)
- **Git**
- **Cloudflare R2 Bucket Configuration**:
  - **Dataset Setup**: You must set up your own dataset following the instructions in the [R2 Dataset Guide](./r2_dataset.md)
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
> - Deploying to multiple servers
> - Managing multi-GPU setups
> - Automating the entire setup process

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
   git clone https://github.com/tplr-ai/templar.git
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
   btcli subnet pow_register --wallet.name default --wallet.hotkey miner --netuid <netuid> --subtensor.network <network>
   ```

6. **Log into Weights & Biases (WandB)**:

   ```bash
   wandb login your_wandb_api_key
   ```

7. **Set Environment Variables**:

   Export necessary environment variables or create a `.env` file in the project root.
   ```bash
   export WANDB_API_KEY=your_wandb_api_key
   export INFLUXDB_TOKEN=your_influxdb_token
   export NODE_TYPE=your_node_type
   export WALLET_NAME=your_wallet_name
   export WALLET_HOTKEY=your_wallet_hotkey
   export CUDA_DEVICE=your_cuda_device
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

   # Dataset R2 credentials - You must set up your own dataset
   # See docs/r2_dataset.md for instructions
   export R2_DATASET_ACCOUNT_ID=your_dataset_account_id
   export R2_DATASET_BUCKET_NAME=your_dataset_bucket_name
   export R2_DATASET_READ_ACCESS_KEY_ID=your_dataset_read_access_key_id
   export R2_DATASET_READ_SECRET_ACCESS_KEY=your_dataset_read_secret_access_key

   # Aggregator R2 credentials
   export R2_AGGREGATOR_ACCOUNT_ID="80f15715bb0b882c9e967c13e677ed7d"
   export R2_AGGREGATOR_BUCKET_NAME="aggregator"
   export R2_AGGREGATOR_READ_ACCESS_KEY_ID="aa7ea943895323963edba7323e4e12b9"
   export R2_AGGREGATOR_READ_SECRET_ACCESS_KEY="8557ba9fc49a5482fff35d591c808bbdf0ec306bf08d9524c2055736aca915d1"
   
   export GITHUB_USER=your_github_username
   ```

8. **Run the Miner**:

   ```bash
   python neurons/miner.py \
     --actual_batch_size 6 \
     --wallet.name default \
     --wallet.hotkey miner \
     --device cuda \
     --use_wandb \
     --netuid <netuid> \
     --subtensor.network <network> \
     --sync_state
   ```

  *PM2 Support Installation
  ```bash
   pm2 start neurons/miner.py --interpreter python3 --name sn3miner -- \
   --actual_batch_size 6 \
   --wallet.name default \
   --wallet.hotkey miner \
   --device cuda \
   --subtensor.network <network> \
   --sync_state \
   --netuid <netuid> 
  ```

---

## Configuration

### Environment Variables

When using Docker Compose, set the following variables in the `docker/.env` file:

```dotenv:docker/.env
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

# GPU Configuration
CUDA_DEVICE=cuda:0

# Additional Settings
DEBUG=false
```

**Note**: The R2 permissions remain unchanged from previous configurations.

### Hardware Requirements

- **GPU Requirements**:
  - Minimum: NVIDIA H100 with 80GB VRAM
- **Storage**: 100GB+ recommended for model and data
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

```
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
