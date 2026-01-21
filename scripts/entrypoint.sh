#!/bin/bash
set -e

# Docker Compose already handles GPU mapping via device_ids
# No need to set CUDA_VISIBLE_DEVICES - Docker provides the correct GPU
echo "Using Docker-mapped GPU configuration"

# Check required environment variables
for var in WALLET_NAME WALLET_HOTKEY NODE_TYPE WANDB_API_KEY NETUID; do
    if [ -z "${!var}" ]; then
        echo "Error: $var environment variable is required"
        exit 1
    fi
done

# Activate virtual environment
source /app/.venv/bin/activate

# Create logs directory
mkdir -p /app/logs

# Check CUDA availability
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "Error: CUDA is not available"
    echo "Available GPUs:"
    (set +e; nvidia-smi -L 2>/dev/null) || echo "nvidia-smi not available"
    exit 1
fi

# Login to wandb non-interactively
wandb login ${WANDB_API_KEY} --relogin

# Convert DEBUG to --debug flag if true
DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
    DEBUG_FLAG="--debug"
fi


# Check CUDA version
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
if [[ "${CUDA_VERSION}" != "12.6" ]]; then
    echo "Warning: Container CUDA version (${CUDA_VERSION}) differs from host CUDA version (12.6)"
fi

# Check NODE_TYPE and start appropriate process
if [ "$NODE_TYPE" = "miner" ]; then
    echo "Starting miner with torchrun..."
    exec torchrun \
        --standalone \
        --nnodes 1 \
        --nproc_per_node 2 \
        neurons/miner.py \
        --wallet.name ${WALLET_NAME} \
        --wallet.hotkey ${WALLET_HOTKEY} \
        --netuid ${NETUID} \
        --device cuda \
        --subtensor.network ${NETWORK} \
        --use_wandb \
        ${PROJECT:+--project ${PROJECT}} \
        ${DEBUG_FLAG}
elif [ "$NODE_TYPE" = "validator" ]; then
    echo "Starting burn validator..."
    exec python neurons/burn.py \
        --wallet.name ${WALLET_NAME} \
        --wallet.hotkey ${WALLET_HOTKEY} \
        --netuid ${NETUID} \
        --subtensor.network ${NETWORK}
elif [ "$NODE_TYPE" = "evaluator" ]; then
    # Count the number of visible GPUs for evaluator
    if [ -n "$NVIDIA_VISIBLE_DEVICES" ]; then
        # Count commas + 1 to get number of GPUs
        NUM_GPUS=$(echo "$NVIDIA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
        NUM_GPUS=$((NUM_GPUS + 1))
    else
        NUM_GPUS=1
    fi
    
    echo "Starting evaluator with torchrun using $NUM_GPUS GPU(s)..."
    exec torchrun \
        --standalone \
        --nnodes 1 \
        --nproc_per_node $NUM_GPUS \
        neurons/evaluator.py \
        --netuid ${NETUID} \
        --device cuda \
        --actual_batch_size ${EVAL_BATCH_SIZE} \
        --tasks "${EVAL_TASKS:-arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag,mmlu}" \
        --eval_interval ${EVAL_INTERVAL:-600} \
        ${CUSTOM_EVAL_PATH:+--custom_eval_path "${CUSTOM_EVAL_PATH}"} \
        ${DEBUG_FLAG}
else
    echo "Error: NODE_TYPE must be \"miner\", \"validator\", or \"evaluator\""
    exit 1
fi 
