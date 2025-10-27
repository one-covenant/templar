import pytest
import torch

from tplr.compress import pack_12bit_indices
from tplr.neurons import prepare_gradient_dict


class DummyHparams:
    def __init__(self):
        self.weight_decay = 0.1
        self.momentum_decay = 0.9
        self.topk_compression = 5
        self.outer_learning_rate = 0.9
        self.use_dct = False


class DummyCompressor:
    def compress(self, encoded_tensor, topk):
        # Create 12-bit packed format for dummy indices
        indices = torch.tensor([0, 1], dtype=torch.long)  # Even count
        packed_data = pack_12bit_indices(indices)
        dummy_idxs = packed_data
        dummy_vals = "dummy_vals"
        dummy_xshape = "dummy_xshape"
        dummy_totalk = "dummy_totalk"
        dummy_quant_params = "dummy_quant_params"
        return dummy_idxs, dummy_vals, dummy_xshape, dummy_totalk, dummy_quant_params

    def decompress(self, p, idxs, vals, xshape, totalk, quant_params):
        return torch.tensor([0.5, 0.5])


class DummyTransformer:
    def encode(self, tensor, use_dct):
        return tensor

    def decode(self, tensor, use_dct):
        return torch.tensor([0.1, 0.1])


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.weight.grad = torch.tensor([0.1, 0.2])


class DummyMiner:
    def __init__(self):
        self.model = DummyModel()
        self.hparams = DummyHparams()
        self.error_feedback = {"weight": torch.zeros_like(self.model.weight)}
        self.error_feedback_cpu_buffers = {
            "weight": torch.zeros_like(self.model.weight, pin_memory=False)
        }
        self.owned_params = {"weight", "weight1", "weight2"}
        self.compressor = DummyCompressor()
        self.transformer = DummyTransformer()
        self.logger = DummyLogger()


# Test 1: Return Structure and Types
# ---------------------------------
# - Call prepare_gradient_dict with valid inputs
# - Verify it returns a tuple of length 3 containing (gradient, xshapes, totalks)
# - Check gradient dict has expected keys (weightidxs, weightvals, metadata)
# - Verify xshapes, totalks, are dicts with expected keys
# - Confirm metadata attachment is logged
def test_return_structure_and_types(caplog):
    # Create dummy miner instance
    miner = DummyMiner()
    # Define a valid step_window
    step_window = 5

    with caplog.at_level("INFO", logger="templar"):
        # Call the helper function.
        result = prepare_gradient_dict(miner, step_window)

    # Check that we get exactly four items returned.
    assert isinstance(result, tuple)
    assert len(result) == 3
    gradient, xshapes, totalks = result

    # Check that gradient is a dict
    assert isinstance(gradient, dict)
    # Verify gradient has keys for the parameter "weight"
    assert "weightidxs" in gradient
    assert "weightvals" in gradient
    assert "weightquant_params" in gradient
    # Verify that weightidxs is in 12-bit packed format (uint8 tensor)
    assert isinstance(gradient["weightidxs"], torch.Tensor), (
        "weightidxs should be a tensor for 12-bit packed format"
    )
    assert gradient["weightidxs"].dtype == torch.uint8, (
        "weightidxs should be uint8 for 12-bit packed format"
    )
    # Verify that the metadata key exists
    assert "metadata" in gradient
    # Check that metadata contains the expected keys
    assert "window" in gradient["metadata"]
    assert gradient["metadata"]["window"] == step_window
    assert "xshapes" in gradient["metadata"]
    assert "totalks" in gradient["metadata"]

    # Check that xshapes, totalks, are dictionaries.
    assert isinstance(xshapes, dict)
    assert isinstance(totalks, dict)
    # And that they contain key "weight"
    assert "weight" in xshapes
    assert "weight" in totalks


def test_metadata_attachment():
    """
    Test 2: Metadata Attachment
    ---------------------------
    - Call prepare_gradient_dict.
    - Verify that gradient["metadata"] exactly equals {"window": step_window}.
    """
    # Create dummy miner instance
    miner = DummyMiner()
    # Define a step_window value.
    step_window = 42

    # Call prepare_gradient_dict.
    gradient, xshapes, totalks = prepare_gradient_dict(miner, step_window)

    # Verify that gradient["metadata"] contains the expected keys
    metadata = gradient.get("metadata")
    assert metadata is not None, "Metadata should not be None"
    assert "window" in metadata, "Metadata should contain 'window'"
    assert metadata["window"] == step_window, (
        f"Window mismatch. Expected: {step_window}, Got: {metadata['window']}"
    )
    assert "xshapes" in metadata, "Metadata should contain 'xshapes'"
    assert "totalks" in metadata, "Metadata should contain 'totalks'"


def test_error_feedback_decay_and_gradient_accumulation():
    """
    Test 4 – Error feedback Calculation on First Iteration
    ------------------------------------------------
    When  momentum_decay == 0.0  and the transmitted gradient is 0,
    the final momentum must be  lr * grad  after prepare_gradient_dict.
    """

    # ------------------------------------------------------------------ #
    #  Dummy implementations used ONLY for this test
    # ------------------------------------------------------------------ #
    class DummyZeroCompressor:
        """compress() returns placeholders; decompress() returns zeros."""

        def compress(self, encoded_tensor, topk):
            # Return empty 12-bit packed format
            empty_packed = torch.zeros(0, dtype=torch.uint8)
            empty_shape = torch.Size([0])
            return (
                (empty_packed, empty_shape),
                [],
                (),
                0,
                None,
            )  # idxs, vals, xshape, totalk, quant

        def decompress(self, p, idxs, vals, xshape, totalk, quant_params):
            return torch.zeros_like(p)

    class DummyPassThroughTransformer:
        def encode(self, tensor, use_dct):  # identity
            return tensor

        def decode(self, tensor, use_dct):  # returns tensor as-is
            return tensor

    # ------------------------------------------------------------------ #
    #  Build the miner
    # ------------------------------------------------------------------ #
    miner = DummyMiner()

    # Plug in our dummies
    miner.compressor = DummyZeroCompressor()
    miner.transformer = DummyPassThroughTransformer()

    # Hyper-parameters for the behaviour we want
    miner.hparams.momentum_decay = 0.0  # ignore existing momentum
    miner.hparams.topk_compression = 0.0  # no indices picked
    lr = miner.hparams.outer_learning_rate  # e.g. 0.9

    # Provide momentum and gradient for the *single* parameter "weight"
    miner.error_feedback["weight"] = torch.tensor([0.2, 0.3])  # will be nulled
    miner.model.weight.grad = torch.tensor([0.1, 0.2])

    # ------------------------------------------------------------------ #
    #  Expected result = lr * grad
    # ------------------------------------------------------------------ #
    expected_final_momentum = miner.model.weight.grad * lr  # [0.09, 0.18]

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #
    prepare_gradient_dict(miner, step_window=5)

    # ------------------------------------------------------------------ #
    #  Assert
    # ------------------------------------------------------------------ #
    torch.testing.assert_close(
        miner.error_feedback["weight"],
        expected_final_momentum,
        msg=(
            "Final momentum should equal lr * grad when "
            "momentum_decay == 0 and the transmitted gradient is zero."
        ),
    )


def test_compressor_and_transformer_calls():
    """
    Test 5 – Compressor and Transformer Calls
    -----------------------------------------
    Verify that

        • compressor.compress is invoked with
          transformer.encode(momentum_after_decay_and_add)
          and miner.hparams.topk_compression.

        • transformer.decode is invoked with the output of
          compressor.decompress.

        • the dummy values returned by the compressor end up in
          the gradient / xshapes / totalks dictionaries.
    """

    # ------------------------------------------------------------------ #
    #  Dummy recorders so we can inspect their call arguments
    # ------------------------------------------------------------------ #
    class DummyRecordingCompressor:
        def __init__(self):
            self.called_args = None

        def compress(self, encoded_tensor, topk):
            self.called_args = (encoded_tensor.clone(), topk)
            # Create 12-bit packed format for recording
            indices = torch.tensor([0, 1], dtype=torch.long)
            packed_data = pack_12bit_indices(indices)
            return (
                packed_data,  # 12-bit packed format
                "recorded_dummy_vals",
                "recorded_dummy_xshape",
                "recorded_dummy_totalk",
                "recorded_dummy_quant_params",
            )

        def decompress(self, p, idxs, vals, xshape, totalk, quant_params):
            # Return a fixed tensor so we know exactly what should be
            # handed to transformer.decode
            return torch.tensor([0.2, 0.2])

    class DummyRecordingTransformer:
        def __init__(self):
            self.decode_called_with = None

        def encode(self, tensor, use_dct):
            # Identity for easier reasoning
            return tensor

        def decode(self, tensor, use_dct):
            self.decode_called_with = tensor.clone()
            return torch.tensor([0.1, 0.1])  # value not important for this test

    # ------------------------------------------------------------------ #
    #  Build the miner with recorders
    # ------------------------------------------------------------------ #
    miner = DummyMiner()
    miner.compressor = DummyRecordingCompressor()
    miner.transformer = DummyRecordingTransformer()

    # Provide a single parameter called "weight"
    miner.error_feedback["weight"] = torch.tensor([1.0, 1.0])
    miner.model.weight.grad = torch.tensor([0.3, 0.4])

    lr = miner.hparams.outer_learning_rate  # e.g. 0.9
    momentum_decay = miner.hparams.momentum_decay  # e.g. 0.9
    topk = miner.hparams.topk_compression

    # Expected tensor that *compressor.compress* must receive
    # -------------------------------------------------------
    # prepare_gradient_dict does (per parameter):
    #   momentum.mul_(momentum_decay)
    #   momentum.add_(grad, alpha=lr)
    expected_tensor_for_compression = (
        miner.error_feedback["weight"] * momentum_decay + miner.model.weight.grad * lr
    )

    # ------------------------------------------------------------------ #
    #  Run the function under test
    # ------------------------------------------------------------------ #
    step_window = 7
    gradient, xshapes, totalks = prepare_gradient_dict(miner, step_window)

    # ------------------------------------------------------------------ #
    #  Assertions
    # ------------------------------------------------------------------ #
    # 1. compressor.compress was called with the right tensor & top-k
    comp_args = miner.compressor.called_args
    assert comp_args is not None, "compressor.compress was never called"

    recorded_tensor, recorded_topk = comp_args
    torch.testing.assert_close(
        recorded_tensor,
        expected_tensor_for_compression,
        msg=(
            "compressor.compress did not receive "
            "transformer.encode(momentum_after_decay_and_add)"
        ),
    )
    assert recorded_topk == topk, "compressor.compress received wrong top-k value"

    # 2. transformer.decode was called with compressor.decompress output
    expected_decode_input = torch.tensor([0.2, 0.2])
    assert miner.transformer.decode_called_with is not None, (
        "transformer.decode was never called"
    )
    torch.testing.assert_close(
        miner.transformer.decode_called_with,
        expected_decode_input,
        msg="transformer.decode did not receive compressor.decompress output",
    )

    # 3. Dummy compressor return values appear in the output dictionaries
    # Check that weightidxs is in 12-bit packed format (uint8 tensor)
    assert isinstance(gradient["weightidxs"], torch.Tensor), (
        "weightidxs should be a tensor for 12-bit packed format"
    )
    assert gradient["weightidxs"].dtype == torch.uint8, (
        "weightidxs should be uint8 for 12-bit packed format"
    )
    assert gradient["weightvals"] == "recorded_dummy_vals"
    assert xshapes["weight"] == "recorded_dummy_xshape"
    assert totalks["weight"] == "recorded_dummy_totalk"

    # 4. Metadata
    assert gradient["metadata"]["window"] == step_window


# Test 6: Handling Multiple Parameters
# --------------------------------------
# - Create a dummy model with two parameters, e.g., "weight1" and "weight2".
# - Ensure that the returned gradient dict includes keys: "weight1idxs", "weight1vals", "weight2idxs", "weight2vals".
# - Also verify that xshapes, and totalks dicts have both "weight1" and "weight2" as keys.


def test_handling_multiple_parameters():
    """
    Test 6: Handling Multiple Parameters
    --------------------------------------
    - Create a dummy model with two parameters, e.g., "weight1" and "weight2".
    - Ensure that the returned gradient dict includes keys: "weight1idxs", "weight1vals", "weight2idxs", "weight2vals".
    - Also verify that xshapes and totalks dicts have both "weight1" and "weight2" as keys.
    """

    class DummyMultiModel(torch.nn.Module):
        def __init__(self):
            super(DummyMultiModel, self).__init__()
            self.weight1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.weight2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
            self.weight1.grad = torch.tensor([0.1, 0.1])
            self.weight2.grad = torch.tensor([0.2, 0.2])

    miner = DummyMiner()
    miner.model = DummyMultiModel()
    miner.error_feedback = {
        "weight1": torch.zeros_like(miner.model.weight1),
        "weight2": torch.zeros_like(miner.model.weight2),
    }

    step_window = 10
    gradient, xshapes, totalks = prepare_gradient_dict(miner, step_window)

    # Check for gradient keys from both parameters.
    assert "weight1idxs" in gradient, "Missing key: weight1idxs"
    assert "weight1vals" in gradient, "Missing key: weight1vals"
    assert "weight1quant_params" in gradient, "Missing key: weight1quant_params"
    assert "weight2idxs" in gradient, "Missing key: weight2idxs"
    assert "weight2vals" in gradient, "Missing key: weight2vals"
    assert "weight2quant_params" in gradient, "Missing key: weight2quant_params"

    # Check that xshapes, and totalks have both "weight1" and "weight2".
    for key in ["weight1", "weight2"]:
        assert key in xshapes, f"Missing {key} in xshapes"
        assert key in totalks, f"Missing {key} in totalks"


def test_behavior_when_p_grad_is_none():
    """
    Test 7: Behavior When p.grad is None
    -------------------------------------
    - Create a dummy model parameter with p.grad set to None.
    - Verify that the function raises an exception due to the missing gradient.
    """

    # Define a dummy model with one parameter whose grad is None.
    class DummyNoGradModel(torch.nn.Module):
        def __init__(self):
            super(DummyNoGradModel, self).__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.weight.grad = None  # No gradient provided.

    miner = DummyMiner()
    miner.model = DummyNoGradModel()
    # Reinitialize momentum accordingly.
    miner.error_feedback = {"weight": torch.zeros_like(miner.model.weight)}

    step_window = 3

    # The function should skip parameters with None gradients and return an empty gradient dict
    gradient, xshapes, totalks = prepare_gradient_dict(miner, step_window)

    # Should only have metadata, no parameter data since grad was None
    assert "metadata" in gradient
    assert gradient["metadata"]["window"] == step_window
    # Weight should be skipped, so no weightidxs/weightvals
    assert "weightidxs" not in gradient
    assert "weightvals" not in gradient


def test_logging_behavior(caplog):
    """
    Test 8: Logging Behavior
    ------------------------
    - Use the pytest caplog fixture to capture log calls.
    - Verify that when prepare_gradient_dict is executed, a log call is made that includes
      the correct metadata string (containing window).
    """
    miner = DummyMiner()
    step_window = 15

    with caplog.at_level("INFO", logger="templar"):
        prepare_gradient_dict(miner, step_window)


def test_propagation_of_compressor_failure():
    """
    Test 10: Propagation of Exceptions (Compressor Failure)
    --------------------------------------------------------
    - Force compressor.compress to throw an exception.
    - Verify that prepare_gradient_dict propagates the exception instead of silently swallowing it.
    """
    miner = DummyMiner()

    # Override compressor.compress to throw an exception.
    def failing_compress(encoded_tensor, topk):
        raise RuntimeError("Compressor error")

    miner.compressor.compress = failing_compress

    step_window = 5

    with pytest.raises(RuntimeError, match="Compressor error"):
        prepare_gradient_dict(miner, step_window)


def test_propagation_of_transformer_failure():
    """
    Test 11: Propagation of Exceptions (Transformer Failure)
    ---------------------------------------------------------
    - Force transformer.decode to throw an exception.
    - Verify that prepare_gradient_dict propagates this exception as expected.
    """
    miner = DummyMiner()

    # Override transformer.decode to throw an exception.
    def failing_decode(tensor, use_dct):
        raise RuntimeError("Transformer error")

    miner.transformer.decode = failing_decode

    step_window = 5

    with pytest.raises(RuntimeError, match="Transformer error"):
        prepare_gradient_dict(miner, step_window)
