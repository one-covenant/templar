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

"""TorchTitan Model Factory

Unified model creation and initialization for TorchTitan models across
evaluator, validator, and miner components.
"""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torchtitan.config import (
    ActivationCheckpoint,
    Float8,
    JobConfig,
    Model,
    Parallelism,
    Training,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3 import (
    Llama3StateDictAdapter,
    TransformerModelArgs,
    llama3_configs,
)
from torchtitan.models.llama3 import Transformer as TitanLlama
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from tqdm.auto import tqdm
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

import tplr
from tplr.distributed import dist_helper


def get_titan_model_args(hparams: SimpleNamespace) -> TransformerModelArgs:
    """Get TorchTitan TransformerModelArgs from predefined configs or custom hparams.

    Args:
        hparams: Hyperparameters object containing model configuration

    Returns:
        TransformerModelArgs configured for TorchTitan

    Raises:
        ValueError: If model_size is not valid
    """
    model_size = hparams.model_size
    sequence_length = hparams.sequence_length
    vocab_size = (
        hparams.tokenizer.vocab_size
        if hasattr(hparams, "tokenizer")
        else hparams.model_config.vocab_size
    )
    # Get tie_embeddings from torchtitan config, default to True for backward compatibility
    tt = getattr(hparams, "torchtitan", SimpleNamespace())
    tie_embeddings = getattr(tt, "tie_embeddings", True)

    if model_size in llama3_configs:
        tplr.logger.info(f"Using predefined TorchTitan config for {model_size}")
        # Create a copy of the config to avoid modifying the original
        args = TransformerModelArgs(**vars(llama3_configs[model_size]))
        # Update max_seq_len with our sequence length
        args.max_seq_len = sequence_length
        # Update vocab_size if provided (from tokenizer)
        args.vocab_size = vocab_size
        args.tie_embeddings = tie_embeddings
        tplr.logger.info(f"Using vocab_size: {vocab_size}")
        return args

    # Fall back to creating custom config from hparams for non-standard sizes (150M, 1B, etc.)
    tplr.logger.info(f"Creating custom TorchTitan config for {model_size} from hparams")

    # Create custom TransformerModelArgs from hparams
    args = TransformerModelArgs(
        dim=hparams.model_config.hidden_size,
        n_layers=hparams.model_config.num_hidden_layers,
        n_heads=hparams.model_config.num_attention_heads,
        n_kv_heads=getattr(
            hparams.model_config,
            "num_key_value_heads",
            hparams.model_config.num_attention_heads,
        ),
        vocab_size=vocab_size,
        ffn_dim_multiplier=None,  # Will calculate from intermediate_size
        multiple_of=256,
        norm_eps=getattr(hparams.model_config, "rms_norm_eps", 1e-5),
        rope_theta=getattr(hparams.model_config, "rope_theta", 10000.0),
        max_seq_len=sequence_length,
        tie_embeddings=tie_embeddings,
    )

    # Calculate ffn_dim_multiplier to match the intermediate_size from hparams
    # The formula in TorchTitan is: ffn_dim = int(8 * dim / 3 * ffn_dim_multiplier)
    # We need to reverse engineer ffn_dim_multiplier from intermediate_size
    target_intermediate_size = hparams.model_config.intermediate_size
    base_ffn = int(8 * args.dim / 3)
    if base_ffn > 0:
        args.ffn_dim_multiplier = target_intermediate_size / base_ffn
    else:
        # Avoid division by zero if dim is 0
        args.ffn_dim_multiplier = 1.0

    tplr.logger.info(
        f"Custom config: dim={args.dim}, n_layers={args.n_layers}, "
        f"n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, "
        f"intermediate_size={target_intermediate_size}"
    )

    return args


def create_job_config(
    hparams: SimpleNamespace, role: Literal["miner", "validator", "evaluator"] = "miner"
) -> JobConfig:
    """Create JobConfig from hparams with role-specific defaults.

    Args:
        hparams: Hyperparameters object containing torchtitan config
        role: Component role (miner, validator, or evaluator)

    Returns:
        JobConfig configured for the specific role
    """
    tt = getattr(hparams, "torchtitan", SimpleNamespace())

    # Role-specific defaults
    mixed_precision_param_default: Literal["bfloat16", "float32"]
    mixed_precision_reduce_default: Literal["float32"] = "float32"  # Only valid value
    compile_default = getattr(tt, "compile", False)

    if role == "evaluator":
        # Evaluator uses minimal settings for inference
        mixed_precision_param_default = "float32"
        enable_cpu_offload_default = False
    elif role == "validator":
        # Validator uses conservative settings
        # Ensure we only get valid values
        mp_param = getattr(tt, "mixed_precision_param", "float32")
        mixed_precision_param_default = (
            "bfloat16" if mp_param == "bfloat16" else "float32"
        )
        enable_cpu_offload_default = False
    else:  # miner
        # Miner uses full settings from config
        # Ensure we only get valid values
        mp_param = getattr(tt, "mixed_precision_param", "float32")
        mixed_precision_param_default = (
            "bfloat16" if mp_param == "bfloat16" else "float32"
        )
        enable_cpu_offload_default = getattr(tt, "enable_cpu_offload", False)

    # Build Training config
    training = Training(
        seq_len=hparams.sequence_length,
        compile=compile_default,
        enable_cpu_offload=enable_cpu_offload_default,
        mixed_precision_param=mixed_precision_param_default,
        mixed_precision_reduce=mixed_precision_reduce_default,
    )

    # Build Parallelism config
    parallelism = Parallelism(
        enable_async_tensor_parallel=getattr(tt, "enable_async_tensor_parallel", False),
        disable_loss_parallel=getattr(tt, "disable_loss_parallel", True),
        fsdp_reshard_after_forward=getattr(tt, "fsdp_reshard_after_forward", "default"),
        enable_compiled_autograd=getattr(tt, "enable_compiled_autograd", False),
    )

    # Build ActivationCheckpoint config
    activation_checkpoint = ActivationCheckpoint(
        mode=tt.activation_checkpoint.get("mode", "selective")
        if hasattr(tt, "activation_checkpoint")
        and isinstance(tt.activation_checkpoint, dict)
        else "selective",
        selective_ac_option=tt.activation_checkpoint.get("option", "op")
        if hasattr(tt, "activation_checkpoint")
        and isinstance(tt.activation_checkpoint, dict)
        else "op",
    )

    return JobConfig(
        training=training,
        parallelism=parallelism,
        model=Model(converters=getattr(tt, "converters", [])),
        float8=Float8(recipe_name=getattr(tt, "float8_recipe_name", None)),
        activation_checkpoint=activation_checkpoint,
    )


def create_parallel_dims(
    world_size: int,
    hparams: SimpleNamespace,
    role: Literal["miner", "validator", "evaluator"] = "miner",
) -> ParallelDims:
    """Create ParallelDims based on role and configuration.

    Args:
        world_size: Number of distributed processes
        hparams: Hyperparameters object containing torchtitan config
        role: Component role (miner, validator, or evaluator)

    Returns:
        ParallelDims configured for the specific role

    Raises:
        ValueError: If parallelization parameters are invalid
    """
    if world_size <= 0:
        world_size = 1
    if role == "evaluator":
        # Evaluator: support both single and multi-GPU configurations
        # Ensure dp_shard is at least 1 to prevent division by zero
        dp_shard = max(1, min(8, world_size))
        if world_size % dp_shard != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"dp_shard degree ({dp_shard})"
            )
        return ParallelDims(
            dp_replicate=world_size // dp_shard,
            dp_shard=dp_shard,
            tp=1,
            pp=1,
            cp=1,
            ep=1,
            world_size=world_size,
        )
    elif role == "validator":
        # Validator: pipeline parallelism with data parallel replication
        # Ensure dp_shard is at least 1 to prevent division by zero
        dp_shard = 4
        if world_size % dp_shard != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"dp_shard degree ({dp_shard})"
            )
        return ParallelDims(
            dp_replicate=world_size // dp_shard,
            dp_shard=dp_shard,
            tp=1,
            pp=1,
            cp=1,
            ep=1,
            world_size=world_size,
        )
    else:  # miner
        # Miner: flexible configuration from hparams
        tt = getattr(hparams, "torchtitan", SimpleNamespace())

        tp_degree = int(getattr(tt, "tp_degree", 1))
        pp_degree = int(getattr(tt, "pp_degree", 1))
        cp_degree = int(getattr(tt, "cp_degree", 1))
        dp_replicate = getattr(tt, "dp_replicate", 1)
        dp_shard = getattr(tt, "dp_shard", 1)

        # Ensure divisors are not zero before coercion to 1 and modulo operations
        if dp_replicate == 0:
            raise ValueError("dp_replicate cannot be zero.")
        if dp_shard == 0:
            raise ValueError("dp_shard cannot be zero.")
        if tp_degree == 0:
            raise ValueError("tp_degree cannot be zero.")
        if pp_degree == 0:
            raise ValueError("pp_degree cannot be zero.")
        if cp_degree == 0:
            raise ValueError("cp_degree cannot be zero.")

        # Coerce to int after zero checks
        dp_replicate = int(dp_replicate)
        dp_shard = int(dp_shard)
        tp_degree = int(tp_degree)
        pp_degree = int(pp_degree)
        cp_degree = int(cp_degree)

        # Validation
        if dp_replicate > 1 and dp_shard > 1:
            raise ValueError(
                "Specify either torchtitan.dp_replicate or torchtitan.dp_shard, "
                "but not both."
            )

        if dp_replicate > 1 and (tp_degree > 1 or pp_degree > 1 or cp_degree > 1):
            raise ValueError("dp_replicate can only be used when tp/pp/cp are all 1.")

        # Ensure world_size is divisible by the product of all parallel degrees
        total_parallel_degree = (
            dp_replicate * dp_shard * tp_degree * pp_degree * cp_degree
        )
        if (
            total_parallel_degree == 0
        ):  # Should be caught by individual zero checks, but as a safeguard
            raise ValueError("Product of parallel degrees cannot be zero.")
        if world_size % total_parallel_degree != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by the product of all parallel degrees "
                f"({dp_replicate}x{dp_shard}x{tp_degree}x{pp_degree}x{cp_degree} = {total_parallel_degree})."
            )

        return ParallelDims(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            tp=tp_degree,
            pp=pp_degree,
            cp=cp_degree,
            ep=1,
            world_size=world_size,
        )


def create_meta_model(
    hparams: SimpleNamespace,
    role: Literal["miner", "validator", "evaluator"] = "miner",
    world_size: int = 1,
) -> nn.Module:
    """Create a TorchTitan model on meta device for fast checkpoint loading.

    This creates the model structure without allocating memory for weights,
    which is useful when you want to load a checkpoint directly.

    Args:
        hparams: Hyperparameters object from tplr.load_hparams()
        role: Component role (miner, validator, or evaluator)
        world_size: Number of distributed processes

    Returns:
        Model on meta device ready for checkpoint loading
    """
    # Get model arguments using predefined TorchTitan configs or custom hparams
    titan_args = get_titan_model_args(hparams)

    # Create parallelization dimensions
    pdims = create_parallel_dims(world_size, hparams, role)

    # Create job config
    job_config = create_job_config(hparams, role)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        # Only build a mesh when distributed is initialized (avoids single-proc errors)
        _ = pdims.build_mesh()

    # Create model on meta device
    with torch.device("meta"):
        model = TitanLlama(titan_args)
        model.args = titan_args

    # Parallelize the model while still on meta device
    model = parallelize_llama(
        model,
        parallel_dims=pdims,
        job_config=job_config,
    )

    return model


def initialize_torchtitan_model(
    hparams: SimpleNamespace,
    role: Literal["miner", "validator", "evaluator"] = "miner",
    device: str = "cuda",
    world_size: int = 1,
) -> nn.Module:
    """Complete model initialization pipeline for TorchTitan models.

    Args:
        hparams: Hyperparameters object from tplr.load_hparams()
        role: Component role (miner, validator, or evaluator)
        device: Device to initialize model on
        world_size: Number of distributed processes

    Returns:
        Initialized and parallelized TorchTitan model
    """
    # Get model arguments using predefined TorchTitan configs or custom hparams
    titan_args = get_titan_model_args(hparams)

    # Create parallelization dimensions
    pdims = create_parallel_dims(world_size, hparams, role)

    # Create job config
    job_config = create_job_config(hparams, role)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        # Only build a mesh when distributed is initialized (avoids single-proc errors)
        _ = pdims.build_mesh()

    # Create model on meta device
    with torch.device("meta"):
        model = TitanLlama(titan_args)
        model.args = titan_args

    # Parallelize the model
    model = parallelize_llama(
        model,
        parallel_dims=pdims,
        job_config=job_config,
    )

    # Initialize weights via rank‑0 broadcast of a single full FP32 state (FSDP2/DCP)
    model.to_empty(device=device)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        rank = dist.get_rank()
        if rank == 0:
            with torch.device("meta"):
                ref = TitanLlama(titan_args)  # unsharded reference
                ref.args = titan_args
            ref.to_empty(device="cpu")
            torch.manual_seed(42)
            with torch.no_grad():
                ref.init_weights()
            full_sd = get_model_state_dict(
                ref, options=StateDictOptions(full_state_dict=True)
            )
        else:
            full_sd = {}
        set_model_state_dict(
            model,
            full_sd,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=True
            ),
        )
        dist.barrier()

    else:
        # Single‑process path (e.g., 1‑GPU validator / CPU evaluator)
        with torch.device("meta"):
            ref = TitanLlama(titan_args)
            ref.args = titan_args
        ref.to_empty(device="cpu")
        torch.manual_seed(42)
        with torch.no_grad():
            ref.init_weights()
        full_sd = get_model_state_dict(
            ref, options=StateDictOptions(full_state_dict=True)
        )
        missing, unexpected = set_model_state_dict(
            model,
            full_sd,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )
        if missing or unexpected:
            raise ValueError(f"Missing {missing}, unexpected {unexpected}")

    # Log initialization details
    if role == "miner":
        tt = getattr(hparams, "torchtitan", SimpleNamespace())
        tp_degree = int(getattr(tt, "tp_degree", 1))
        dp_shard = int(getattr(tt, "dp_shard", 1))
        tplr.logger.info(
            f"[Model Factory] Initialized {role} model with TP={tp_degree}, "
            f"DP_shard={dp_shard}, world_size={world_size}"
        )
    else:
        tplr.logger.info(f"[Model Factory] Initialized {role} model on {device}")

    return model


def initialize_weights_inplace(
    model: nn.Module,
    hparams: SimpleNamespace,
    world_size: int = 1,
) -> None:
    """Initialize weights in-place on an existing model that was created on meta device.

    This function is used when we need to initialize a model that was already created
    and moved to device but doesn't have weights yet (e.g., when no checkpoint is loaded).

    Args:
        model: Model that needs weight initialization (already on device)
        hparams: Hyperparameters object
        world_size: Number of distributed processes
    """
    # Get model arguments
    titan_args = get_titan_model_args(hparams)

    if dist_helper.is_distributed():
        if dist_helper.is_master:
            # Create reference model on CPU for weight initialization
            with torch.device("meta"):
                ref = TitanLlama(titan_args)
                ref.args = titan_args
            ref.to_empty(device="cpu")
            torch.manual_seed(42)
            with torch.no_grad():
                ref.init_weights()
            full_sd = get_model_state_dict(
                ref, options=StateDictOptions(full_state_dict=True)
            )
        else:
            full_sd = {}

        # Broadcast and set weights
        set_model_state_dict(
            model,
            full_sd,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=True
            ),
        )
        dist_helper.safe_barrier("weight_init", dist_helper.local_rank)
    else:
        # Single-process path
        with torch.device("meta"):
            ref = TitanLlama(titan_args)
            ref.args = titan_args
        ref.to_empty(device="cpu")
        torch.manual_seed(42)
        with torch.no_grad():
            ref.init_weights()
        full_sd = get_model_state_dict(
            ref, options=StateDictOptions(full_state_dict=True)
        )
        set_model_state_dict(
            model,
            full_sd,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )


def _get_unwrapped_model(model: nn.Module) -> "TitanLlama":
    """Recursively unwraps a model from DDP or FSDP wrappers."""
    while hasattr(model, "module"):
        model = model.module
    if not isinstance(model, TitanLlama):
        raise ValueError(
            f"Expected model to be a TitanLlama instance, got {type(model)} instead."
        )
    return model


def _get_actual_intermediate_size(
    titan_state_dict: "OrderedDict[str, torch.Tensor]", hparams: SimpleNamespace
) -> int:
    """Extract actual intermediate size from model state dict.

    Args:
        titan_state_dict: TorchTitan model state dict
        hparams: Hyperparameters object

    Returns:
        Actual intermediate size from model or hparams fallback
    """
    for key in [
        "layers.0._orig_mod.feed_forward.w1.weight",
        "layers.0.feed_forward.w1.weight",
    ]:
        if key in titan_state_dict:
            return titan_state_dict[key].shape[0]

    # Fallback to hparams
    tplr.logger.warning(
        f"Could not determine intermediate_size from model, using hparams value: {hparams.model_config.intermediate_size}"
    )
    return hparams.model_config.intermediate_size


def _get_hf_config_from_titan(
    titan_model: nn.Module,
    hparams: SimpleNamespace,
    titan_state_dict: "OrderedDict[str, torch.Tensor]",
) -> LlamaConfig:
    """Builds a HuggingFace LlamaConfig from a TorchTitan model."""
    unwrapped_titan_model = _get_unwrapped_model(titan_model)
    titan_args = unwrapped_titan_model.args
    if not isinstance(titan_args, TransformerModelArgs):
        raise ValueError(
            f"Expected unwrapped model to have TransformerModelArgs, "
            f"got {type(titan_args)} instead."
        )

    actual_intermediate_size = _get_actual_intermediate_size(titan_state_dict, hparams)
    tplr.logger.info(f"Using intermediate_size: {actual_intermediate_size}")

    return LlamaConfig(
        vocab_size=titan_args.vocab_size,
        hidden_size=titan_args.dim,
        intermediate_size=actual_intermediate_size,
        num_hidden_layers=titan_args.n_layers,
        num_attention_heads=titan_args.n_heads,
        num_key_value_heads=titan_args.n_kv_heads,
        hidden_act=getattr(hparams.model_config, "hidden_act", "silu"),
        max_position_embeddings=hparams.sequence_length,
        initializer_range=getattr(hparams.model_config, "initializer_range", 0.02),
        rms_norm_eps=getattr(hparams.model_config, "rms_norm_eps", 1e-5),
        use_cache=False,
        rope_theta=getattr(hparams.model_config, "rope_theta", 10000.0),
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        pretraining_tp=1,
        tie_word_embeddings=False,
    )


def convert_titan_to_hf(
    titan_model: nn.Module,
    hparams: SimpleNamespace,
    save_path: str | None = None,
    model_args: dict[str, Any] | None = None,
    is_master: bool = False,
) -> None:
    """Convert TorchTitan model to HuggingFace format.

    This function converts a TorchTitan Llama model to HuggingFace format,
    handling state dict conversion and proper configuration mapping.

    Args:
        titan_model: TorchTitan model to convert
        hparams: Hyperparameters object containing model configuration
        save_path: Optional path to save the converted HuggingFace model

    Returns:
        None

    Raises:
        ValueError: If conversion fails
    """
    try:
        # Get TorchTitan state dict
        tplr.logger.info("Getting TorchTitan state dict")
        titan_state_dict = get_model_state_dict(
            titan_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,  # Automatically offload to CPU
            ),
        )

        if is_master:
            if model_args:
                hf_config = LlamaConfig(**model_args)

            elif isinstance(titan_model, TitanLlama):
                # Directly get config from TitanLlama model if possible
                tplr.logger.info("Using TorchTitan model args for HuggingFace config")
                hf_config = _get_hf_config_from_titan(
                    titan_model, hparams, titan_state_dict
                )
                tplr.logger.info(
                    "Finished creating HuggingFace config from TorchTitan models"
                )

            else:
                tplr.logger.info(
                    "Using hparams.model_config to create HuggingFace config"
                )
                hf_config = LlamaConfig(
                    vocab_size=hparams.model_config.vocab_size,
                    hidden_size=hparams.model_config.hidden_size,
                    intermediate_size=_get_actual_intermediate_size(
                        titan_state_dict, hparams
                    ),
                    num_hidden_layers=hparams.model_config.num_hidden_layers,
                    num_attention_heads=hparams.model_config.num_attention_heads,
                    num_key_value_heads=getattr(
                        hparams.model_config,
                        "num_key_value_heads",
                        hparams.model_config.num_attention_heads,
                    ),
                    hidden_act=getattr(hparams.model_config, "hidden_act", "silu"),
                    max_position_embeddings=hparams.sequence_length,
                )

            # Create HuggingFace model
            hf_model = LlamaForCausalLM(hf_config)

            # Clean state dict (remove prefixes and special keys, and convert DTensors)
            cleaned = {}
            for k, v in titan_state_dict.items():
                if "freqs_cis" in k:
                    continue
                nk = k.replace("_orig_mod.", "").replace("module.", "")
                # Convert DTensor to full tensor
                if isinstance(v, DTensor):
                    if not v.is_cuda:
                        raise ValueError(
                            "DTensor must be on CUDA device for allgather. Try self.model.to(self.device)"
                        )
                    v = v.full_tensor()
                cleaned[nk] = v

            # Use official adapter for state dict conversion
            unwrapped_titan_model = _get_unwrapped_model(titan_model)
            titan_args = unwrapped_titan_model.args
            if not isinstance(titan_args, TransformerModelArgs):
                raise ValueError(
                    "Expected unwrapped model to have TransformerModelArgs, "
                    f"got {type(titan_args)} instead."
                )
            adapter = Llama3StateDictAdapter(titan_args)
            hf_state_dict = adapter.to_hf(cleaned)

            # Load converted state dict into HuggingFace model
            hf_model.load_state_dict(hf_state_dict, strict=is_master)

            # Save if path provided
            if save_path:
                tplr.logger.info(
                    "Saving model state. For large models, this takes a while"
                )

                hf_model.save_pretrained(save_path)
                tplr.logger.info(
                    f"Successfully saved TorchTitan model in HuggingFace format to {save_path}"
                )

        return

    except Exception as e:
        tplr.logger.error(f"HF conversion failed: {e}", exc_info=True)
        raise ValueError(
            f"Failed to convert TorchTitan model to HuggingFace format: {e}"
        )
