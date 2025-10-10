# Validator Setup

This document provides a comprehensive guide on how to set up and run a validator using `validator.py`. Validators are crucial components of **τemplar**, responsible for evaluating miners' contributions by assessing their uploaded gradients.

## Table of Contents

- [Validator Setup](#validator-setup)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Using Docker Compose (Recommended)](#using-docker-compose-recommended)
    - [Manual Installation](#manual-installation)
  - [Running the Validator](#running-the-validator)
    - [Using Docker Compose](#using-docker-compose)
    - [Running Without Docker](#running-without-docker)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Hardware Requirements](#hardware-requirements)
    - [Network Options](#network-options)
    - [InfluxDB Configuration](#influxdb-configuration)
  - [Monitoring](#monitoring)
    - [Logs](#logs)
    - [Performance](#performance)
  - [Troubleshooting](#troubleshooting)
  - [Validator Operations](#validator-operations)
    - [State Synchronization](#state-synchronization)
    - [Evaluation Process](#evaluation-process)
    - [Weight Setting](#weight-setting)

---

## Introduction

This guide will help you set up and run a validator for **τemplar**. Validators play a critical role in maintaining the integrity of the network by evaluating miners' contributions and updating weights accordingly.

---

## Prerequisites

- **NVIDIA GPU** with CUDA support
  - **Minimum required: 4x H200 GPUs**
- **Ubuntu** (or Ubuntu-based Linux distribution)
- **Docker** and **Docker Compose**
- **Git**
- **Python 3.12+** (for manual installation)
- **Hugging Face Authentication**:
  - Create a Hugging Face account and generate a token at https://huggingface.co/settings/tokens
  - Accept the Gemma model terms at https://huggingface.co/google/gemma-3-270m (required for tokenizer access)
  - Set `HF_TOKEN` environment variable with your token
- **Cloudflare R2 Bucket Configuration**:
  - **Dataset Setup**: Please refer to [Shared Sharded Dataset Documentation](./shared_sharded_dataset.md) for complete dataset setup instructions, including:
    - R2 bucket settings
    - Dataset download process
  - **Gradient Bucket Setup**:
    1. **Create a Bucket**: Name it the same as your **account ID** and set the **region** to **ENAM**.
    2. **Generate Tokens**:
       - **Read Token**: Admin Read permissions.
       - **Write Token**: Admin Read & Write permissions.
    3. **Store Credentials**: You'll need these for the `.env` file.

---

## Installation

### Using Docker Compose (Recommended)

1. **Install Docker and Docker Compose**:

   Follow the same steps as in the [Miner Setup](#using-docker-compose-recommended) section.

2. **Enable Docker GPU Support**:

   Follow the official NVIDIA Container Toolkit installation guide:

   ```bash
   # 1. Configure the production repository
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # 2. Update package listings
   sudo apt-get update

   # 3. Install the NVIDIA Container Toolkit
   sudo apt-get install -y nvidia-container-toolkit

   # 4. Configure Docker runtime
   sudo nvidia-ctk runtime configure --runtime=docker

   # 5. Restart Docker daemon
   sudo systemctl restart docker

   # 6. Test GPU support
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

   If you see the `nvidia-smi` output, GPU support is working correctly.

   For detailed instructions and other Linux distributions, refer to the [official NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

3. **Clone the Repository**:

   ```bash
   git clone https://github.com/one-covenant/templar.git
   cd templar
   ```

4. **Navigate to the Docker Directory**:

   ```bash
   cd docker
   ```

5. **Create and Populate the `.env` File**:

   Create a `.env` file in the `docker` directory by copying the `.env.example`:

   ```bash
   cp .env.example .env
   ```

   Populate the `.env` file with your configuration. Variables to set:

   ```dotenv:docker/.env
   # Required: Hugging Face token for tokenizer access
   HF_TOKEN=<your_huggingface_token>
   
   # Add your Weights & Biases API key
   WANDB_API_KEY=<your_wandb_api_key>


   # Cloudflare R2 Credentials - Add your R2 credentials below
   R2_GRADIENTS_ACCOUNT_ID=<your_r2_account_id>
   R2_GRADIENTS_BUCKET_NAME=<your_r2_bucket_name>

   R2_GRADIENTS_READ_ACCESS_KEY_ID=<your_r2_read_access_key_id>
   R2_GRADIENTS_READ_SECRET_ACCESS_KEY=<your_r2_read_secret_access_key>

   R2_GRADIENTS_WRITE_ACCESS_KEY_ID=<your_r2_write_access_key_id>
   R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY=<your_r2_write_secret_access_key>

   # Dataset R2 credentials - See docs/shared_sharded_dataset.md for instructions
   R2_DATASET_ACCOUNT_ID=<your_dataset_account_id>
   R2_DATASET_BUCKET_NAME=<your_dataset_bucket_name>
   R2_DATASET_READ_ACCESS_KEY_ID=<your_dataset_read_access_key_id>
   R2_DATASET_READ_SECRET_ACCESS_KEY=<your_dataset_read_secret_access_key>

   # Aggregator R2 credentials
   R2_AGGREGATOR_ACCOUNT_ID=8af7f92a8a0661cf7f1ac0420c932980
   R2_AGGREGATOR_BUCKET_NAME=aggregator
   R2_AGGREGATOR_READ_ACCESS_KEY_ID=bb4b9f02a64dacead181786b8f353b67
   R2_AGGREGATOR_READ_SECRET_ACCESS_KEY=f50761d0fbb0773c55f61debdf87439735c32c096fe4b1ab6aa6bfb7f52aa30b

   # Wallet Configuration
   WALLET_NAME=<your_wallet_name>
   WALLET_HOTKEY=<your_wallet_hotkey>

   # Network Configuration
   NETWORK=finney
   NETUID=3
   # GPU Configuration (automatically handled by Docker)
   # Validator service uses GPUs 0, 1, and 2 from the host
   # Node Type
   NODE_TYPE=validator
   # Additional Settings
   DEBUG=false
   ```

   **Note**: Set `NODE_TYPE` to `validator`.

6. **Update `docker-compose.yml`**:

   Ensure that the `docker-compose.yml` file is correctly configured for your setup.

7. **Run Docker Compose**:

   Start the validator using Docker Compose:

   ```bash
   docker compose -f docker/compose.yml up -d
   ```

### Manual Installation

If you prefer to run the validator without Docker, follow the instructions in the [Running Without Docker](#running-without-docker) section.

---

## Running the Validator

### Using Docker Compose

After completing the installation steps, your validator should be running. Check it with:

```bash
docker ps
```

You should see a container named `templar-validator-<WALLET_HOTKEY>`.

### Running Without Docker

1. **Install System Dependencies**:

   Same as in the miner setup.

2. **Install NVIDIA CUDA Drivers**:

   Install the appropriate NVIDIA CUDA drivers.

3. **Clone the Repository**:

   ```bash
   git clone https://github.com/one-covenant/templar.git
   cd templar
   ```

4. **Set Up Python Environment**:

   ```bash
   export HF_TOKEN=your_huggingface_token  # Required for tokenizer access
   export WANDB_API_KEY=your_wandb_api_key
   export NODE_TYPE=your_node_type
   export WALLET_NAME=your_wallet_name
   export WALLET_HOTKEY=your_wallet_hotkey
   # GPU is automatically assigned by Docker (GPUs 0,1,2 for validator)
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

   # Dataset R2 credentials - See docs/shared_sharded_dataset.md for instructions
   export R2_DATASET_ACCOUNT_ID=your_dataset_account_id
   export R2_DATASET_BUCKET_NAME=your_dataset_bucket_name
   export R2_DATASET_READ_ACCESS_KEY_ID=your_dataset_read_access_key_id
   export R2_DATASET_READ_SECRET_ACCESS_KEY=your_dataset_read_secret_access_key

   # Aggregator R2 credentials
   export R2_AGGREGATOR_ACCOUNT_ID=8af7f92a8a0661cf7f1ac0420c932980
   export R2_AGGREGATOR_BUCKET_NAME=aggregator
   export R2_AGGREGATOR_READ_ACCESS_KEY_ID=bb4b9f02a64dacead181786b8f353b67
   export R2_AGGREGATOR_READ_SECRET_ACCESS_KEY=f50761d0fbb0773c55f61debdf87439735c32c096fe4b1ab6aa6bfb7f52aa30b
   
   export GITHUB_USER=your_github_username
   ```

5. **Create and Register Validator Wallet**:

   ```bash
   # Create coldkey if not already created
   btcli wallet new_coldkey --wallet.name default --n-words 12

   # Create and register validator hotkey
   btcli wallet new_hotkey --wallet.name default --wallet.hotkey validator --n-words 12
   btcli subnet pow_register --wallet.name default --wallet.hotkey validator --netuid <netuid> --subtensor.network <network>
   ```

6. **Log into Weights & Biases (WandB)**:

   ```bash
   wandb login your_wandb_api_key
   ```

7. **Set Environment Variables**:

   Export necessary environment variables as in the miner setup.

8. **Run the Validator**:

   ```bash
   torchrun --standalone --nnodes 1 --nproc_per_node 4 \
     neurons/validator.py \
     --wallet.name <wallet_name> \
     --wallet.hotkey <hotkey> \
     --device cuda \
     --netuid 3 \
     --subtensor.network <network> \
     --use_wandb
   ```

---

## Configuration

### Environment Variables

Set the following in the `docker/.env` file when using Docker Compose:

```dotenv:docker/.env
# Required: Hugging Face token for tokenizer access
HF_TOKEN=your_huggingface_token

WANDB_API_KEY=your_wandb_api_key
INFLUXDB_TOKEN=your_influxdb_token

# Cloudflare R2 Credentials
R2_ACCOUNT_ID=your_r2_account_id

R2_READ_ACCESS_KEY_ID=your_r2_read_access_key_id
R2_READ_SECRET_ACCESS_KEY=your_r2_read_secret_access_key

R2_WRITE_ACCESS_KEY_ID=your_r2_write_access_key_id
R2_WRITE_SECRET_ACCESS_KEY=your_r2_write_secret_access_key

# Additional Gradient R2 credentials
R2_GRADIENTS_ACCOUNT_ID=your_r2_gradients_account_id
R2_GRADIENTS_BUCKET_NAME=your_r2_gradients_bucket_name

R2_GRADIENTS_READ_ACCESS_KEY_ID=your_r2_gradients_read_access_key_id
R2_GRADIENTS_READ_SECRET_ACCESS_KEY=your_r2_gradients_read_secret_access_key

R2_GRADIENTS_WRITE_ACCESS_KEY_ID=your_r2_gradients_write_access_key_id
R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY=your_r2_gradients_write_secret_access_key

# Dataset R2 credentials - See docs/shared_sharded_dataset.md for instructions
R2_DATASET_ACCOUNT_ID=your_dataset_account_id
R2_DATASET_BUCKET_NAME=your_dataset_bucket_name
R2_DATASET_READ_ACCESS_KEY_ID=your_dataset_read_access_key_id
R2_DATASET_READ_SECRET_ACCESS_KEY=your_dataset_read_secret_access_key

# Aggregator R2 credentials
R2_AGGREGATOR_ACCOUNT_ID=8af7f92a8a0661cf7f1ac0420c932980
R2_AGGREGATOR_BUCKET_NAME=aggregator
R2_AGGREGATOR_READ_ACCESS_KEY_ID=bb4b9f02a64dacead181786b8f353b67
R2_AGGREGATOR_READ_SECRET_ACCESS_KEY=f50761d0fbb0773c55f61debdf87439735c32c096fe4b1ab6aa6bfb7f52aa30b

# Wallet Configuration
WALLET_NAME=default
WALLET_HOTKEY=your_validator_hotkey_name

# Network Configuration
NETWORK=finney
NETUID=3

# GPU Configuration (automatically handled by Docker)
# Validator service uses GPUs 0, 1, and 2 from the host

# Node Type
NODE_TYPE=validator

# Additional Settings
DEBUG=false
```

**Note**: The R2 permissions remain unchanged.

### Hardware Requirements

- **GPU Requirements**:
  - **Minimum required: 4x H200 GPUs** (as defined in min_compute.yml)
  - **Minimum CPU**: 64 cores, 3.5 GHz (recommended: 128 cores, 4.0 GHz)
  - **Minimum RAM**: 1200 GB (recommended: 1500 GB)
  - **Minimum Network**: 1024 Mbps download/upload bandwidth
- **Storage**: 1000GB minimum, 2000GB recommended for model and evaluation data
- **Network**: High-bandwidth, stable connection for state synchronization

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
  docker logs -f templar-validator-${WALLET_HOTKEY}
  ```

- **Weights & Biases**:

  - Ensure `--use_wandb` is enabled
  - Monitor evaluation metrics and network statistics

### Performance

Key metrics to monitor:

- GPU utilization
- Memory usage
- Network bandwidth
- Evaluation throughput
- Weight setting frequency

---

## Troubleshooting

- **State Synchronization Failures**: Check network settings and ensure the validator is properly registered and connected.
- **Out of Memory Errors**: Reduce `--actual_batch_size`.
- **Network Connectivity Issues**: Verify firewall settings and network configurations.

---

## Validator Operations

### State Synchronization

- The validator synchronizes its model with the latest global state.
- It gathers and applies gradients from miners to maintain consistency.

### Evaluation Process

1. **Collect Miner Gradients**: Gathers compressed gradients submitted by miners.
2. **Evaluate Contributions**: Assesses the impact of each miner's gradient on model performance.
3. **Compute Scores**: Calculates scores based on loss improvement.
4. **Update Weights**: Adjusts miners' weights on the blockchain accordingly.

### Weight Setting

- **Scoring Mechanism**: Based on the performance improvement contributed by miners.
- **Update Frequency**: Weights are periodically updated on the blockchain.
- **Impact**: Influences reward distribution and miner reputation in the network.
