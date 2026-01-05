# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Global imports
import json
from pathlib import Path
from types import SimpleNamespace

from torchtitan.models.llama3 import llama3_configs
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama import LlamaConfig

# Local imports
from .logging import logger

DEFAULT_HPARAMS = {
    # Run configuration
    "spec_version": 5,
    "project": "templar",
    # Model parameters
    "sequence_length": 1024,
    "pages_per_window": 2,
    "batch_size": 8,
    "learning_rate": 0.001,
    # Window and sync parameters
    "blocks_per_window": 2,
    "windows_per_weights": 10,
    # Optimization parameters
    "momentum_decay": 0.999,
    "topk_compression": 32,
    "target_chunk": 64,
    "scores_alpha": 0.001,
    # Model architecture (these should be in your hparams.json)
    "tokenizer_name": "huggyllama/llama-7b",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "num_key_value_heads": 32,
    "activation_function": "silu",
    "max_position_embeddings": 2048,
    # Bucket configuration
    "bucket_name": "your-default-bucket-name",
    # Scheduler parameters
    "warmup_steps": 250,
    "alpha_f": 0.1,  # Final learning rate multiplier
    "t_max": 20000,  # Total steps for cosine decay
    "outer_steps_per_shard": 455,  # 1024**3 * 100 / (2048*192*30*20)
}


def create_namespace(hparams: dict) -> SimpleNamespace:
    """
    Create a SimpleNamespace from the hyperparameters and add model configuration.

    Args:
        hparams (dict): Hyperparameters dictionary.

    Returns:
        SimpleNamespace: Namespace containing hyperparameters and model configuration.
    """
    # Merge with defaults
    full_hparams = DEFAULT_HPARAMS.copy()
    full_hparams.update(hparams)

    hparams_ns = SimpleNamespace(**full_hparams)

    # ── expose torchtitan sub‑dict as its own namespace ──────────────
    if isinstance(full_hparams.get("torchtitan"), dict):
        hparams_ns.torchtitan = SimpleNamespace(**full_hparams["torchtitan"])

    # Initialize tokenizer
    try:
        # Check for HF token in environment for gated models
        import os

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            logger.info(
                f"Using HF token from environment for {hparams_ns.tokenizer_name}"
            )
        hparams_ns.tokenizer = AutoTokenizer.from_pretrained(
            hparams_ns.tokenizer_name,
            verbose=False,
            clean_up_tokenization_spaces=True,
            token=token,  # Will be None if not set, which is fine for public models
        )
        hparams_ns.tokenizer.pad_token = hparams_ns.tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Initialize model config
    try:
        hparams_ns.model_config = LlamaConfig(
            vocab_size=hparams_ns.tokenizer.vocab_size,
            hidden_size=hparams_ns.hidden_size,
            num_hidden_layers=hparams_ns.num_hidden_layers,
            num_attention_heads=hparams_ns.num_attention_heads,
            intermediate_size=hparams_ns.intermediate_size,
            num_key_value_heads=hparams_ns.num_key_value_heads,
            activation_function=hparams_ns.activation_function,
            max_position_embeddings=hparams_ns.max_position_embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to create model config: {e}")
        raise

    return hparams_ns


def load_hparams(
    hparams_dir: str = "hparams", use_local_run_hparams: bool = False
) -> SimpleNamespace:
    """
    Load hyperparameters by discovering the model_size from the base hparams.json.

    The merge order is:
    1. `DEFAULT_HPARAMS` (lowest priority)
    2. Base `hparams.json` (which defines the model_size)
    3. Model-specific `[model_size].json`
    4. `hparams-local-run.json` (if requested, highest priority)

    Args:
        hparams_dir (str): The directory containing the hyperparameter files.
        use_local_run_hparams (bool): If True, override with `hparams-local-run.json`.

    Returns:
        SimpleNamespace: A namespace containing the final hyperparameters and model config.
    """
    hparams = DEFAULT_HPARAMS.copy()
    hparams_dir_path = Path(hparams_dir)
    base_hparams_file = hparams_dir_path / "hparams.json"

    # 1. Load the base hparams file first to discover the model_size
    try:
        with open(base_hparams_file, "r") as f:
            base_hparams = json.load(f)
            hparams.update(base_hparams)
        logger.info(f"Loaded base config from {base_hparams_file}")
    except FileNotFoundError:
        logger.error(f"CRITICAL: Base config file not found at {base_hparams_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading {base_hparams_file}: {e}")
        raise

    # 2. Get model_size from the loaded hparams and load the specific file
    model_size = hparams.get("model_size")
    if not model_size:
        raise ValueError(
            f"CRITICAL: 'model_size' must be defined in {base_hparams_file}"
        )

    # Check if this is a standard TorchTitan model size
    if model_size in llama3_configs:
        # For TorchTitan sizes, we don't need a model-specific JSON file
        logger.info(f"Using TorchTitan predefined config for {model_size}")
    else:
        # For custom sizes (150M, 1B, etc.), load the model-specific JSON file
        model_hparams_file = hparams_dir_path / f"{model_size}.json"
        try:
            with open(model_hparams_file, "r") as f:
                model_specific_hparams = json.load(f)
                hparams.update(model_specific_hparams)
            logger.info(f"Loaded model-specific config from {model_hparams_file}")
        except FileNotFoundError:
            logger.error(f"Model-specific config file not found: {model_hparams_file}")
            raise ValueError(f"Could not find hparams for model_size '{model_size}'")
        except Exception as e:
            logger.error(f"Error loading {model_hparams_file}: {e}")
            raise

    # 3. (Optional) Load and merge local run overrides
    if use_local_run_hparams:
        local_run_file = hparams_dir_path / "hparams-local-run.json"
        try:
            with open(local_run_file, "r") as f:
                local_run_hparams = json.load(f)
                hparams.update(local_run_hparams)
            logger.info(
                f"Loaded and applied local run overrides from {local_run_file}: {local_run_hparams}"
            )
        except FileNotFoundError:
            logger.warning(f"Local run specified but {local_run_file} not found.")
        except Exception as e:
            logger.error(f"Error loading {local_run_file}: {e}")
            raise

    logger.info(
        f"Final project: '{hparams.get('project')}', Model size: '{hparams.get('model_size')}'"
    )
    return create_namespace(hparams)
