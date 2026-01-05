import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torchtitan.models.llama3 import Transformer as TitanLlama
from torchtitan.models.llama3 import TransformerModelArgs
from transformers import LlamaConfig, LlamaForCausalLM

from tplr.model_factory import (
    _get_hf_config_from_titan,
    _get_unwrapped_model,
    convert_titan_to_hf,
    create_parallel_dims,
    initialize_torchtitan_model,
)


def setup_distributed_environment() -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


setup_distributed_environment()


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.cuda.set_device", return_value=None)
class TestModelFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs):
        setup_distributed_environment()

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_get_unwrapped_model(
        self,
        mock_get_world_size,
        mock_is_initialized,
        mock_cuda_available,
        mock_set_device,
    ):
        """Test that the model unwrapping utility works correctly."""
        # Create a mock model
        hparams = SimpleNamespace(
            model_size="150M",
            sequence_length=128,
            model_config=SimpleNamespace(
                vocab_size=10,
                num_hidden_layers=1,
                hidden_size=10,
                num_attention_heads=1,
                intermediate_size=20,
            ),
        )
        inner_model = initialize_torchtitan_model(
            hparams, role="evaluator", world_size=1, device="cpu"
        )
        # Test that the unwrapped model is the original model
        with patch("tplr.model_factory.isinstance", return_value=True):
            unwrapped = _get_unwrapped_model(inner_model)
            self.assertIs(unwrapped, inner_model)

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_convert_titan_to_hf_with_titan_model(
        self,
        mock_get_world_size,
        mock_is_initialized,
        mock_cuda_available,
        mock_set_device,
    ):
        """Test conversion from a Titan model to a HuggingFace model."""
        # Mock hparams
        hparams = SimpleNamespace(
            model_size="150M",
            sequence_length=128,
            model_config=SimpleNamespace(
                vocab_size=100,
                num_hidden_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=128,
                hidden_act="silu",
                initializer_range=0.02,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
            ),
        )

        # Mock Titan model and its args
        titan_model = initialize_torchtitan_model(
            hparams, role="evaluator", world_size=1, device="cpu"
        )

        with patch(
            "tplr.model_factory._get_actual_intermediate_size", return_value=128
        ):
            with patch("tplr.model_factory.Llama3StateDictAdapter.to_hf") as mock_to_hf:
                # Create a dummy hf_state_dict that matches the hf_model
                hf_config = LlamaConfig(
                    vocab_size=100,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                )
                hf_model_for_state_dict = LlamaForCausalLM(hf_config)
                hf_state_dict = hf_model_for_state_dict.state_dict()
                mock_to_hf.return_value = hf_state_dict

                # Convert the model
                hf_model = convert_titan_to_hf(titan_model, hparams, is_master=True)

                # Assertions
                self.assertIsNone(hf_model)

    def test_convert_titan_to_hf_with_model_args(
        self, mock_cuda_available, mock_set_device
    ):
        """Test that provided model_args override model's own args."""
        hparams = SimpleNamespace()
        titan_model = MagicMock(spec=TitanLlama)
        titan_model.args = TransformerModelArgs(
            vocab_size=10, n_layers=1, dim=10, n_heads=1
        )
        # Create a dummy state_dict with at least one key
        titan_model.state_dict.return_value = {"dummy_key": torch.zeros(1)}

        model_args = {
            "vocab_size": 200,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        }

        with patch("tplr.model_factory.Llama3StateDictAdapter.to_hf") as mock_to_hf:
            # Create a dummy hf_state_dict that matches the hf_model
            hf_config = LlamaConfig(**model_args)
            hf_model_for_state_dict = LlamaForCausalLM(hf_config)
            hf_state_dict = hf_model_for_state_dict.state_dict()
            mock_to_hf.return_value = hf_state_dict

            # Convert the model
            hf_model = convert_titan_to_hf(
                titan_model, hparams, model_args=model_args, is_master=True
            )

            # Assertions
            self.assertIsNone(hf_model)

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_get_hf_config_from_titan(
        self,
        mock_get_world_size,
        mock_is_initialized,
        mock_cuda_available,
        mock_set_device,
    ):
        """Test the helper function for creating HF config from a Titan model."""
        hparams = SimpleNamespace(
            model_size="150M",
            sequence_length=128,
            model_config=SimpleNamespace(
                vocab_size=100,
                num_hidden_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=128,
                hidden_act="silu",
            ),
        )
        titan_model = initialize_torchtitan_model(
            hparams, role="evaluator", world_size=1, device="cpu"
        )
        state_dict = titan_model.state_dict()

        with patch(
            "tplr.model_factory._get_actual_intermediate_size", return_value=128
        ):
            config = _get_hf_config_from_titan(titan_model, hparams, state_dict)
            self.assertIsInstance(config, LlamaConfig)
            self.assertEqual(config.vocab_size, 100)


class TestCreateParallelDims(unittest.TestCase):
    def test_evaluator_role(self):
        hparams = SimpleNamespace()
        pdims = create_parallel_dims(world_size=1, hparams=hparams, role="evaluator")
        self.assertEqual(pdims.dp_replicate, 1)
        self.assertEqual(pdims.dp_shard, 1)
        self.assertEqual(pdims.tp, 1)
        self.assertEqual(pdims.pp, 1)
        self.assertEqual(pdims.cp, 1)
        self.assertEqual(pdims.ep, 1)
        self.assertEqual(pdims.world_size, 1)

        pdims = create_parallel_dims(world_size=4, hparams=hparams, role="evaluator")
        self.assertEqual(pdims.dp_replicate, 1)
        self.assertEqual(pdims.dp_shard, 4)
        self.assertEqual(pdims.tp, 1)
        self.assertEqual(pdims.pp, 1)
        self.assertEqual(pdims.cp, 1)
        self.assertEqual(pdims.ep, 1)
        self.assertEqual(pdims.world_size, 4)

    def test_validator_role(self):
        hparams = SimpleNamespace()
        pdims = create_parallel_dims(world_size=4, hparams=hparams, role="validator")
        self.assertEqual(pdims.dp_replicate, 1)
        self.assertEqual(pdims.dp_shard, 4)
        self.assertEqual(pdims.tp, 1)
        self.assertEqual(pdims.pp, 1)
        self.assertEqual(pdims.cp, 1)
        self.assertEqual(pdims.ep, 1)
        self.assertEqual(pdims.world_size, 4)

    def test_miner_role_default(self):
        hparams = SimpleNamespace(torchtitan=SimpleNamespace())
        pdims = create_parallel_dims(world_size=1, hparams=hparams, role="miner")
        self.assertEqual(pdims.dp_replicate, 1)
        self.assertEqual(pdims.dp_shard, 1)
        self.assertEqual(pdims.tp, 1)
        self.assertEqual(pdims.pp, 1)
        self.assertEqual(pdims.cp, 1)
        self.assertEqual(pdims.ep, 1)
        self.assertEqual(pdims.world_size, 1)

    def test_miner_role_custom_tp(self):
        hparams = SimpleNamespace(torchtitan=SimpleNamespace(tp_degree=2))
        pdims = create_parallel_dims(world_size=2, hparams=hparams, role="miner")
        self.assertEqual(pdims.tp, 2)
        self.assertEqual(pdims.world_size, 2)

    def test_miner_role_invalid_dp_replicate_and_dp_shard(self):
        hparams = SimpleNamespace(
            torchtitan=SimpleNamespace(dp_replicate=2, dp_shard=2)
        )
        with self.assertRaisesRegex(
            ValueError,
            "Specify either torchtitan.dp_replicate or torchtitan.dp_shard, but not both.",
        ):
            create_parallel_dims(world_size=4, hparams=hparams, role="miner")

    def test_miner_role_invalid_dp_replicate_with_tp(self):
        hparams = SimpleNamespace(
            torchtitan=SimpleNamespace(dp_replicate=2, tp_degree=2)
        )
        with self.assertRaisesRegex(
            ValueError, "dp_replicate can only be used when tp/pp/cp are all 1."
        ):
            create_parallel_dims(world_size=4, hparams=hparams, role="miner")

    def test_miner_role_world_size_not_divisible_by_dp(self):
        hparams = SimpleNamespace(torchtitan=SimpleNamespace(dp_shard=2))
        with self.assertRaisesRegex(
            ValueError,
            "world_size .* must be divisible by the product of all parallel degrees",
        ):
            create_parallel_dims(world_size=3, hparams=hparams, role="miner")

    def test_miner_role_world_size_not_divisible_by_tp(self):
        hparams = SimpleNamespace(torchtitan=SimpleNamespace(tp_degree=2))
        with self.assertRaisesRegex(
            ValueError,
            "world_size .* must be divisible by the product of all parallel degrees",
        ):
            create_parallel_dims(world_size=3, hparams=hparams, role="miner")

    def test_zero_dp_replicate_or_dp_shard(self):
        hparams_zero_dp_replicate = SimpleNamespace(
            torchtitan=SimpleNamespace(dp_replicate=0)
        )
        with self.assertRaisesRegex(ValueError, "dp_replicate cannot be zero."):
            create_parallel_dims(
                world_size=1, hparams=hparams_zero_dp_replicate, role="miner"
            )

        hparams_zero_dp_shard = SimpleNamespace(torchtitan=SimpleNamespace(dp_shard=0))
        with self.assertRaisesRegex(ValueError, "dp_shard cannot be zero."):
            create_parallel_dims(
                world_size=1, hparams=hparams_zero_dp_shard, role="miner"
            )

    def test_zero_tp_degree(self):
        hparams_zero_tp_degree = SimpleNamespace(
            torchtitan=SimpleNamespace(tp_degree=0)
        )
        with self.assertRaisesRegex(ValueError, "tp_degree cannot be zero."):
            create_parallel_dims(
                world_size=1, hparams=hparams_zero_tp_degree, role="miner"
            )


if __name__ == "__main__":
    unittest.main()
