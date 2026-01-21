# Templar Documentation

> **⚠️ 100% Burn Period Active**: Miners are not being evaluated and no emissions are being distributed. See the main [README](../README.md) for current status and information about Templar: Crusades.

Welcome to the Templar documentation. This index provides an overview of all available documentation to help you navigate and find the information you need.

## 📚 Table of Contents

### Getting Started
- **[Miner Setup](./miner.md)** - Complete guide for setting up and running a miner
- **[Validator Setup](./validator.md)** - Instructions for deploying and operating a validator
- **[Dataset Setup](./r2_dataset.md)** - Overview of dataset requirements and setup options
  - [FineWeb-edu Dataset](./r2_dataset-fineweb-edu.md) - Legacy dataset setup guide
  - [DCLM Dataset](./r2_dataset-dclm.md) - Current dataset preparation guide

### Deployment Options
- **[Ansible Setup](./miner-setup-ansible.md)** - Automated deployment using Ansible playbooks
- **[Localnet/Testnet](./localnet.md)** - Local development environment setup

### Tools & Scripts
- **[Model Evaluator](./evaluator.md)** - Autonomous model evaluation service
- **[Model Converter](./model_converter.md)** - GGUF format conversion service
- **[R2 Bucket Management](./r2_bucket_management.md)** - Cloudflare R2 storage utilities

### Infrastructure
- **[Log Archives](./log_archive.md)** - Accessing and analyzing network logs

## 🚀 Quick Links

### For Miners
1. [Dataset Setup](./r2_dataset.md) - Set up your dataset first
2. [Miner Setup](./miner.md) - Deploy your miner
3. [Ansible Alternative](./miner-setup-ansible.md) - Automated deployment option

### For Validators
1. [Validator Setup](./validator.md) - Deploy your validator
2. [Evaluator Service](./evaluator.md) - Optional evaluation service

### For Developers
1. [Localnet Setup](./localnet.md) - Local testing environment
2. [Model Converter](./model_converter.md) - Model format conversion
3. [Log Archives](./log_archive.md) - Debug and analyze logs

## 📖 Documentation Overview

### Core Components

#### Miner Documentation
- **[miner.md](./miner.md)**: Comprehensive guide covering Docker and manual installation methods
- **[miner-setup-ansible.md](./miner-setup-ansible.md)**: Automated deployment for multiple hosts

#### Validator Documentation
- **[validator.md](./validator.md)**: Complete validator deployment and operation guide

#### Dataset Documentation
- **[r2_dataset.md](./r2_dataset.md)**: Main dataset overview and selection guide
- **[r2_dataset-dclm.md](./r2_dataset-dclm.md)**: DCLM dataset (current dataset)
- **[r2_dataset-fineweb-edu.md](./r2_dataset-fineweb-edu.md)**: Legacy FineWeb-edu dataset setup

### Development Tools

#### Testing & Development
- **[localnet.md](./localnet.md)**: Local Bittensor network for development

#### Model Tools
- **[evaluator.md](./evaluator.md)**: Automated benchmark evaluation
- **[model_converter.md](./model_converter.md)**: GGUF format conversion

### Infrastructure & Utilities

#### Storage Management
- **[r2_bucket_management.md](./r2_bucket_management.md)**: R2 bucket utilities

#### Monitoring & Logs
- **[log_archive.md](./log_archive.md)**: Access historical network logs

## 🔧 Prerequisites

Before diving into specific documentation:

1. **System Requirements**:
   - **NVIDIA GPU with CUDA support (H200 required - 141GB VRAM minimum)**
   - Ubuntu 22.04+ or compatible Linux
   - Python 3.11+
   - Docker & Docker Compose (for containerized deployment)
   - **CPU**: 32 cores, 3.5 GHz minimum
   - **RAM**: 800 GB minimum
   - **Network**: 1024 Mbps download/upload bandwidth minimum

2. **Network Requirements**:
   - Registered Bittensor wallet
   - Stable internet connection
   - Sufficient bandwidth for dataset operations

3. **Storage Requirements**:
   - Cloudflare R2 account and buckets
   - Adequate local storage for models and data
