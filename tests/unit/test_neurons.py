import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from tplr.neurons import (
    catchup_with_aggregation_server,
    compare_model_with_debug_dict,
    determine_slash_egregiousness,
    instantiate_slashing_multiplier,
    outer_step,
    prepare_gradient_dict,
    update_peers,
)


class _FakeComms:
    def __init__(self):
        self.peers = []
        self.reserve_peers = []
        self.get_peer_list = AsyncMock()
        self.get = AsyncMock()
        self.gather = AsyncMock()
        self.gradient_timestamp = AsyncMock()


class TestSlashingUtils(unittest.TestCase):
    def test_determine_slash_egregiousness(self):
        self.assertEqual(determine_slash_egregiousness(0.4), "high")
        self.assertEqual(determine_slash_egregiousness(0.5), "max")
        self.assertEqual(determine_slash_egregiousness(0.6), "mega")
        with self.assertRaises(ValueError):
            determine_slash_egregiousness(1.1)

    def test_instantiate_slashing_multiplier(self):
        multipliers = instantiate_slashing_multiplier()
        self.assertIn("high", multipliers)
        self.assertIn("max", multipliers)
        self.assertIn("mega", multipliers)
        self.assertEqual(multipliers["high"], 0.5)
        self.assertEqual(multipliers["max"], 0.0)
        self.assertEqual(multipliers["mega"], 0.0)


class TestCompareModelWithDebugDict(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        # Create a proper parameter-like object with all required attributes
        self.param1 = torch.nn.Parameter(
            torch.tensor([1.0, 2.0, 3.0]), requires_grad=False
        )
        self.model.named_parameters.return_value = [("param1", self.param1)]

    def test_compare_model_with_debug_dict_perfect_match(self):
        # Debug dict should contain last 2 elements to match to_local() behavior
        debug_dict = {"param1_debug": [2.0, 3.0]}
        result = asyncio.run(
            compare_model_with_debug_dict(self.model, debug_dict, 0.01)
        )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["l2_norm"], 0.0)
        self.assertAlmostEqual(result["avg_steps_behind"], 0.0)

    @pytest.mark.skip(
        reason="TODO: Fix NaN issue in avg_steps_behind calculation with mocked parameters"
    )
    def test_compare_model_with_debug_dict_mismatch(self):
        # Debug dict should contain last 2 elements (but with different values)
        debug_dict = {"param1_debug": [2.2, 3.3]}
        result = asyncio.run(
            compare_model_with_debug_dict(self.model, debug_dict, 0.01)
        )
        self.assertTrue(result["success"])
        self.assertGreater(result["l2_norm"], 0.0)
        self.assertGreater(result["avg_steps_behind"], 0.0)


class TestUpdatePeers(unittest.TestCase):
    def setUp(self):
        self.instance = MagicMock()
        self.instance.comms = _FakeComms()  # Use _FakeComms
        self.instance.comms.peers = []
        self.instance.next_peers = None
        self.instance.peers_update_window = 0
        self.instance.hparams.peer_replacement_frequency = 10
        self.instance.hparams.peer_list_window_margin = 2

    def test_update_peers_initial_fetch(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Directly set the return value for the AsyncMock
        self.instance.comms.get_peer_list.return_value = ([1, 2, 3], [4, 5], 1)

        loop.run_until_complete(update_peers(self.instance, 1, 0.0))

        self.instance.comms.get_peer_list.assert_called_with(fetch_previous=True)
        loop.close()

    def test_update_peers_scheduled_update(self):
        self.instance.comms.peers = [1, 2, 3]
        self.instance.peers_update_window = 1
        self.instance.next_peers = [4, 5, 6]
        self.instance.next_reserve_peers = [7, 8]

        asyncio.run(update_peers(self.instance, 1, 0.0))

        self.assertEqual(self.instance.comms.peers, [4, 5, 6])
        self.assertEqual(self.instance.comms.reserve_peers, [7, 8])
        self.assertIsNone(self.instance.next_peers)


class TestOuterStep(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.transformer = MagicMock()
        self.compressor = MagicMock()
        self.xshapes = {"param1": (10,)}
        self.totalks = {"param1": 2}
        self.device = "cpu"
        self.wandb_run = MagicMock()

        # Mock model parameters with actual tensor
        self.param1 = torch.nn.Parameter(torch.zeros(10))
        self.param1.grad = None
        self.model.named_parameters.return_value = [("param1", self.param1)]
        self.model.module.named_parameters.return_value = [("param1", self.param1)]
        self.model.module.state_dict.return_value = {"param1": torch.tensor([1.0, 2.0])}

    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_outer_step_master_node(
        self, mock_is_initialized, mock_get_world_size, mock_broadcast
    ):
        gather_result = SimpleNamespace(
            uids=["peer1"],
            global_steps=[1],
            state_dict=SimpleNamespace(
                param1idxs=[torch.tensor([1, 2])],
                param1vals=[torch.tensor([0.1, 0.2])],
                param1quant_params=[None],
            ),
        )

        self.compressor.batch_decompress.return_value = torch.rand(10)
        self.compressor.maybe_dequantize_values.return_value = [
            torch.tensor([0.1, 0.2])
        ]
        self.transformer.decode.return_value = torch.rand(10)

        outer_step(
            self.model,
            self.optimizer,
            gather_result=gather_result,
            transformer=self.transformer,
            compressor=self.compressor,
            xshapes=self.xshapes,
            totalks=self.totalks,
            device=self.device,
            is_master=True,
            world_size=2,
            use_dct=False,
            wandb_run=self.wandb_run,
            global_step=1,
        )

        self.optimizer.step.assert_called_once()
        self.wandb_run.log.assert_called()
        self.assertTrue(mock_broadcast.called)

    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_outer_step_worker_node(
        self, mock_is_initialized, mock_get_world_size, mock_broadcast
    ):
        outer_step(
            self.model,
            self.optimizer,
            gather_result=None,
            transformer=self.transformer,
            compressor=self.compressor,
            xshapes=self.xshapes,
            totalks=self.totalks,
            device=self.device,
            is_master=False,
            world_size=2,
            use_dct=False,
        )

        self.optimizer.step.assert_not_called()
        self.assertTrue(mock_broadcast.called)


class TestPrepareGradientDict(unittest.TestCase):
    def setUp(self):
        self.miner = MagicMock()
        self.miner.hparams.outer_learning_rate = 0.01
        self.miner.hparams.momentum_decay = 0.9
        self.miner.hparams.use_dct = False
        self.miner.hparams.topk_compression = 0.1
        self.miner.model = MagicMock()
        self.miner.owned_params = {"param1", "param2"}
        self.miner.error_feedback = {
            "param1": torch.zeros(10),
            "param2": torch.zeros(10),
        }
        self.miner.error_feedback_cpu_buffers = {
            "param1": torch.zeros(10, pin_memory=False),
            "param2": torch.zeros(10, pin_memory=False),
        }
        self.miner.transformer = MagicMock()
        self.miner.compressor = MagicMock()

        # Mock model parameters and gradients
        self.param1 = MagicMock()
        self.param1.grad = torch.rand(10)
        self.param1.device = "cpu"
        self.param2 = MagicMock()
        self.param2.grad = torch.rand(10)
        self.param2.device = "cpu"

        self.miner.model.named_parameters.return_value = [
            ("param1", self.param1),
            ("param2", self.param2),
        ]

        # Mock transformer and compressor outputs
        self.miner.transformer.encode.return_value = torch.rand(10)
        self.miner.compressor.compress.return_value = (
            torch.tensor([1, 2]),
            torch.tensor([0.1, 0.2]),
            (10,),
            2,
            None,
        )
        self.miner.compressor.decompress.return_value = torch.rand(10)
        self.miner.transformer.decode.return_value = torch.rand(10)

    def test_prepare_gradient_dict_happy_path(self):
        gradient, xshapes, totalks = prepare_gradient_dict(self.miner, 1)

        self.assertIn("param1idxs", gradient)
        self.assertIn("param1vals", gradient)
        self.assertIn("param2idxs", gradient)
        self.assertIn("param2vals", gradient)
        self.assertIn("metadata", gradient)
        self.assertEqual(gradient["metadata"]["window"], 1)
        self.assertEqual(self.miner.transformer.encode.call_count, 2)
        self.assertEqual(self.miner.compressor.compress.call_count, 2)


class TestCatchupWithAggregationServer(unittest.TestCase):
    def setUp(self):
        # Mock instance
        self.instance = MagicMock()
        self.instance.start_window = 0
        self.instance.current_window = 5

        # Initialize comms with _FakeComms instance
        self.instance.comms = _FakeComms()
        self.instance.comms.metagraph = MagicMock()
        self.instance.comms.metagraph.S = torch.tensor(
            [0.1, 0.9]
        )  # Set S tensor on the fake metagraph

        self.instance.model = MagicMock()
        self.instance.outer_optimizer = MagicMock()

        # Patch torch.cuda.is_available to avoid RuntimeError in CI
        self.cuda_patch = patch("torch.cuda.is_available", return_value=False)
        self.cuda_patch.start()
        self.instance.transformer = MagicMock()
        self.instance.compressor = MagicMock()
        self.instance.xshapes = {}
        self.instance.totalks = {}
        self.instance.config.device = "cpu"
        self.instance.hparams.use_dct = False
        self.instance.hparams.inner_steps = 10
        self.instance.hparams.time_window_delta_seconds = 10
        self.instance.local_rank = 0  # Add local_rank for dist_helper.safe_barrier
        self.instance.global_step = 0  # Add global_step for tracking outer steps
        self.instance.loop = MagicMock()
        self.instance.loop.run_in_executor = AsyncMock(return_value=12345)
        self.instance.query_block_timestamp = MagicMock(return_value=12345)

    def tearDown(self):
        self.cuda_patch.stop()

    @patch("tplr.distributed.dist_helper.safe_barrier")
    @patch("tplr.distributed.dist_helper.broadcast")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="gloo")  # Mock get_backend
    @patch("tplr.neurons.outer_step")
    @patch("tplr.neurons.compare_model_with_debug_dict")
    def test_catchup_happy_path(
        self,
        mock_compare,
        mock_outer_step,
        mock_get_backend,
        mock_is_initialized,
        mock_broadcast,
        mock_barrier,
    ):
        # Arrange
        self.instance.current_window = 3
        # Create proper response objects that match what outer_step expects
        # The state_dict should contain param + "idxs" and param + "vals" keys
        mock_get = AsyncMock()
        # Set up multiple return values for the different get calls
        mock_get.side_effect = [
            # Window 1 aggregator fetch
            SimpleNamespace(
                success=True,
                data={
                    "state_dict": {
                        "param1idxs": torch.tensor([0, 1]),
                        "param1vals": torch.tensor([0.1, 0.2]),
                    },
                    "uids": [0, 1],
                },
            ),
            # Window 1 debug dict fetch
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1_debug": torch.rand(10)}, "uids": [0, 1]},
            ),
            # Window 2 aggregator fetch
            SimpleNamespace(
                success=True,
                data={
                    "state_dict": {
                        "param1idxs": torch.tensor([2, 3]),
                        "param1vals": torch.tensor([0.3, 0.4]),
                    },
                    "uids": [0, 1],
                },
            ),
            # Window 2 debug dict fetch
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1_debug": torch.rand(10)}, "uids": [0, 1]},
            ),
        ]
        self.instance.comms.get = mock_get
        mock_compare.return_value = {
            "success": True,
            "avg_steps_behind": 0.1,
            "l2_norm": 0.1,
        }

        # Act
        asyncio.run(catchup_with_aggregation_server(self.instance, 0))

        # Assert
        self.assertEqual(mock_outer_step.call_count, 2)  # Windows 1 and 2
        self.assertEqual(
            self.instance.comms.get.call_count, 4
        )  # Called for aggregator and debug dict
        mock_broadcast.assert_called()
        mock_barrier.assert_called()

    @patch("tplr.distributed.dist_helper.safe_barrier")
    @patch("tplr.distributed.dist_helper.broadcast")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="gloo")  # Mock get_backend
    @patch("tplr.neurons.outer_step")
    def test_catchup_fallback_to_live_gather(
        self,
        mock_outer_step,
        mock_get_backend,
        mock_is_initialized,
        mock_broadcast,
        mock_barrier,
    ):
        # Arrange
        self.instance.current_window = 2
        mock_get = AsyncMock(return_value=SimpleNamespace(success=False, data=None))
        self.instance.comms.get = mock_get
        # Create a proper SimpleNamespace for state_dict that won't interfere with mocking
        state_dict_ns = SimpleNamespace()
        state_dict_ns.param1 = torch.rand(10)
        gather_result = SimpleNamespace(state_dict=state_dict_ns)
        self.instance.comms.gather = AsyncMock(return_value=gather_result)

        # Act
        asyncio.run(catchup_with_aggregation_server(self.instance, 0))

        # Assert
        self.assertEqual(
            self.instance.comms.get.call_count, 2
        )  # Called for aggregator and debug dict
        self.instance.comms.gather.assert_called_once()
        mock_outer_step.assert_called_once()
        mock_broadcast.assert_called()
        mock_barrier.assert_called()

    @patch("tplr.distributed.dist_helper.safe_barrier")
    @patch("tplr.distributed.dist_helper.broadcast")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="gloo")  # Mock get_backend
    @patch("tplr.neurons.outer_step")
    def test_catchup_failed_fallback(
        self,
        mock_outer_step,
        mock_get_backend,
        mock_is_initialized,
        mock_broadcast,
        mock_barrier,
    ):
        # Arrange
        self.instance.current_window = 2
        mock_get = AsyncMock(return_value=SimpleNamespace(success=False, data=None))
        self.instance.comms.get = mock_get
        self.instance.comms.gather = AsyncMock(return_value=None)  # Gather fails

        # Act
        asyncio.run(catchup_with_aggregation_server(self.instance, 0))

        # Assert
        self.assertEqual(self.instance.comms.get.call_count, 1)
        self.instance.comms.gather.assert_called_once()
        mock_outer_step.assert_not_called()
        mock_broadcast.assert_called()
        # mock_barrier.assert_called() # Barrier is skipped in this scenario

    @patch("tplr.distributed.dist_helper.safe_barrier")
    @patch("tplr.distributed.dist_helper.broadcast")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="gloo")  # Mock get_backend
    @patch("tplr.neurons.outer_step")
    @patch("tplr.neurons.compare_model_with_debug_dict")
    def test_catchup_skipped_window(
        self,
        mock_compare,
        mock_outer_step,
        mock_get_backend,
        mock_is_initialized,
        mock_broadcast,
        mock_barrier,
    ):
        # Arrange
        self.instance.current_window = 4
        mock_get = AsyncMock()
        mock_get.side_effect = [
            SimpleNamespace(success=False, data=None),  # Fail for window 1
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1": torch.rand(10)}, "uids": [0, 1]},
            ),  # Success for window 2
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1": torch.rand(10)}, "uids": [0, 1]},
            ),  # Success for window 3
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1": torch.rand(10)}, "uids": [0, 1]},
            ),  # debug dict for w2
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1": torch.rand(10)}, "uids": [0, 1]},
            ),  # debug dict for w3
        ]
        self.instance.comms.get = mock_get
        mock_compare.return_value = {
            "success": True,
            "avg_steps_behind": 0.1,
            "l2_norm": 0.1,
        }

        # Act
        asyncio.run(catchup_with_aggregation_server(self.instance, 0))

        # Assert
        self.assertEqual(self.instance.comms.get.call_count, 5)
        self.assertEqual(mock_outer_step.call_count, 2)  # Should only step on success
        self.assertEqual(
            mock_broadcast.call_count, 5
        )  # Two per successful window + one for failed window
        mock_barrier.assert_called()

    @patch("tplr.distributed.dist_helper.safe_barrier")
    @patch("tplr.distributed.dist_helper.broadcast")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="gloo")  # Mock get_backend
    @patch("tplr.neurons.outer_step")
    @patch("tplr.neurons.compare_model_with_debug_dict")
    def test_chain_progression(
        self,
        mock_compare,
        mock_outer_step,
        mock_get_backend,
        mock_is_initialized,
        mock_broadcast,
        mock_barrier,
    ):
        # Arrange
        self.instance.current_window = 2

        async def get_side_effect(*args, **kwargs):
            # Simulate chain progressing
            if kwargs.get("key") == "aggregator" and self.instance.current_window < 5:
                self.instance.current_window += 1
            return SimpleNamespace(
                success=True,
                data={"state_dict": {"param1": torch.rand(10)}, "uids": [0, 1]},
            )

        mock_get = AsyncMock(side_effect=get_side_effect)
        self.instance.comms.get = mock_get
        mock_compare.return_value = {
            "success": True,
            "avg_steps_behind": 0.1,
            "l2_norm": 0.1,
        }

        # Act
        asyncio.run(catchup_with_aggregation_server(self.instance, 0))

        # Assert
        self.assertEqual(self.instance.comms.get.call_count, 8)
        self.assertEqual(mock_outer_step.call_count, 4)
        self.assertEqual(
            mock_broadcast.call_count, 8
        )  # Two per window iteration (has_debug_dict + comparison)
        mock_barrier.assert_called()

    @patch("tplr.distributed.dist_helper.safe_barrier")
    @patch("tplr.distributed.dist_helper.broadcast")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_backend", return_value="gloo")  # Mock get_backend
    @patch("tplr.neurons.outer_step")
    @patch("tplr.neurons.compare_model_with_debug_dict")
    def test_malformed_payload(
        self,
        mock_compare,
        mock_outer_step,
        mock_get_backend,
        mock_is_initialized,
        mock_broadcast,
        mock_barrier,
    ):
        # Arrange
        self.instance.current_window = 3
        mock_get = AsyncMock()
        mock_get.side_effect = [
            SimpleNamespace(success=True, data={}),  # Malformed payload for window 1
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1": torch.rand(10)}, "uids": [0, 1]},
            ),  # Success for window 2
            SimpleNamespace(
                success=True,
                data={"state_dict": {"param1": torch.rand(10)}, "uids": [0, 1]},
            ),  # debug dict for w2
        ]
        self.instance.comms.get = mock_get
        mock_compare.return_value = {
            "success": True,
            "avg_steps_behind": 0.1,
            "l2_norm": 0.1,
        }

        # Act
        asyncio.run(catchup_with_aggregation_server(self.instance, 0))

        # Assert
        self.assertEqual(self.instance.comms.get.call_count, 3)
        self.assertEqual(mock_outer_step.call_count, 1)  # Should only step on success
        self.assertEqual(
            mock_broadcast.call_count, 3
        )  # One for failed window, two for successful window
        mock_barrier.assert_called()


if __name__ == "__main__":
    unittest.main()
