import math

import pytest
import torch
import torch.nn as nn

from tplr.neurons import compare_model_with_debug_dict


class SimpleModel(nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def setup_model():
    """Create a model with deterministic weights for testing."""
    model = SimpleModel()

    # Set deterministic weights for reproducible tests
    with torch.no_grad():
        # Set specific values for the first layer
        model.linear1.weight.fill_(0.1)
        model.linear1.bias.fill_(0.01)

        # Set specific values for the second layer
        model.linear2.weight.fill_(0.2)
        model.linear2.bias.fill_(0.02)

    return model


@pytest.mark.asyncio
async def test_exact_match(setup_model):
    """Test when model parameters exactly match debug dict values."""
    model = setup_model

    # Create debug dict that exactly matches the model parameters
    debug_dict = {}
    for name, param in model.named_parameters():
        debug_dict[name + "_debug"] = param.flatten()[:2].detach().cpu().tolist()

    learning_rate = 0.01

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    assert result["max_diff"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_steps_behind"] == pytest.approx(0.0, abs=1e-6)
    assert result["max_steps_behind"] == pytest.approx(0.0, abs=1e-6)
    assert result["param_count"] > 0
    assert result["learning_rate"] == 0.01


@pytest.mark.asyncio
async def test_one_step_behind(setup_model):
    """Test when model parameters are one step behind debug dict values."""
    model = setup_model
    learning_rate = 0.01

    # Create debug dict with values that are one step ahead
    debug_dict = {}
    for name, param in model.named_parameters():
        # Make debug values one learning_rate step ahead
        values = param.flatten()[:2].detach().cpu()
        ahead_values = values + learning_rate
        debug_dict[name + "_debug"] = ahead_values.tolist()

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    # Each parameter should be exactly one step behind
    assert result["avg_steps_behind"] == pytest.approx(1.0, abs=1e-2)
    assert result["max_steps_behind"] == pytest.approx(1.0, abs=1e-2)


@pytest.mark.asyncio
async def test_multiple_steps_behind(setup_model):
    """Test when model parameters are multiple steps behind debug dict values."""
    model = setup_model
    learning_rate = 0.01
    steps_behind = 5.0

    # Create debug dict with values that are multiple steps ahead
    debug_dict = {}
    for name, param in model.named_parameters():
        # Make debug values multiple learning_rate steps ahead
        values = param.flatten()[:2].detach().cpu()
        ahead_values = values + (learning_rate * steps_behind)
        debug_dict[name + "_debug"] = ahead_values.tolist()

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["avg_steps_behind"] == pytest.approx(steps_behind, abs=1e-2)
    assert result["max_steps_behind"] == pytest.approx(steps_behind, abs=1e-2)


@pytest.mark.asyncio
async def test_missing_parameters(setup_model):
    """Test with a debug dict missing some parameters."""
    model = setup_model

    # Create debug dict with only one parameter
    debug_dict = {}
    first_param = next(iter(model.named_parameters()))
    name, param = first_param
    debug_dict[name + "_debug"] = param.flatten()[:2].detach().cpu().tolist()

    learning_rate = 0.01

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    # Should only count the parameters that were found in debug_dict
    assert result["param_count"] == 2  # Two values from the first parameter
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_empty_debug_dict(setup_model):
    """Test with an empty debug dict."""
    model = setup_model
    debug_dict = {}
    learning_rate = 0.01

    # Compare model with debug dict
    result = await compare_model_with_debug_dict(model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["param_count"] == 0
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert math.isinf(result["avg_l2_norm"])
    assert math.isinf(result["avg_steps_behind"])


@pytest.mark.asyncio
async def test_different_devices():
    """Test comparing model on different devices if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test")

    # Create model on CPU
    cpu_model = SimpleModel()

    # Create debug dict from CPU model
    debug_dict = {}
    for name, param in cpu_model.named_parameters():
        debug_dict[name + "_debug"] = param.flatten()[:2].detach().cpu().tolist()

    # Move model to CUDA
    cuda_model = SimpleModel().to("cuda")

    # Copy weights from CPU model to ensure they match
    with torch.no_grad():
        for (_, cpu_param), (_, cuda_param) in zip(
            cpu_model.named_parameters(), cuda_model.named_parameters()
        ):
            cuda_param.copy_(cpu_param)

    learning_rate = 0.01

    # Compare CUDA model with CPU debug dict
    result = await compare_model_with_debug_dict(cuda_model, debug_dict, learning_rate)

    # Verify the results
    assert result["success"] is True
    assert result["l2_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["avg_steps_behind"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_custom_index_range(setup_model):
    """Test that the function uses the specified index_range for parameter comparison."""
    model = setup_model
    learning_rate = 0.01

    # Modify specific indices of model parameters to have different values
    with torch.no_grad():
        # Check if parameter is large enough for our test
        param = model.linear1.weight.data.flatten()
        if param.numel() > 7:
            # Set indices 0-2 to 0.1 (default)
            param[0:2] = 0.1
            # Set indices 5-7 to 0.5 (different value)
            param[5:7] = 0.5

    # Create debug dict with exact matches for both default and custom indices
    debug_dict = {}
    for name, param in model.named_parameters():
        param_flat = param.flatten()
        num_elements = param_flat.numel()
        debug_dict[name + "_debug"] = param_flat[:2].detach().cpu().tolist()

    # Test with default indices (0, 2) - should be an exact match
    default_result = await compare_model_with_debug_dict(
        model, debug_dict, learning_rate
    )

    # Should show exact match because we're checking default indices
    assert default_result["avg_steps_behind"] == pytest.approx(0.0, abs=1e-6)

    # Skip further tests if the parameters aren't large enough
    first_param = next(iter(model.parameters()))
    if first_param.numel() <= 7:
        pytest.skip("Model parameters too small for advanced index testing")

    # Create a new debug dict for the custom indices test
    custom_debug_dict = {}
    for name, param in model.named_parameters():
        param_flat = param.flatten()
        num_elements = param_flat.numel()

        # Only include parameters with enough elements for the custom range
        if num_elements >= 7:
            # Include values that will match the custom range too
            values = param_flat[5:7].detach().cpu().tolist()
            custom_debug_dict[name + "_debug"] = values

    # Test with a custom range that should match
    matching_result = await compare_model_with_debug_dict(
        model, custom_debug_dict, learning_rate, index_range=(5, 7)
    )

    # Should be an exact match
    assert matching_result["param_count"] > 0  # Ensure we actually compared something
    assert matching_result["avg_steps_behind"] == pytest.approx(0.0, abs=1e-6)

    mismatched_debug_dict = {}
    for name, param in model.named_parameters():
        if name == "linear1.weight":
            param_flat = param.flatten()
            if param_flat.numel() >= 7:
                # mismatch values
                mismatched_debug_dict[name + "_debug"] = [0.3, 0.3]

    # Test with custom indices (5, 7) - should detect the difference
    mismatched_result = await compare_model_with_debug_dict(
        model, mismatched_debug_dict, learning_rate, index_range=(5, 7)
    )

    # Verify we actually compared parameters
    assert mismatched_result["param_count"] > 0

    # Calculate expected difference and verify it matches actual difference
    expected_diff = (
        0.5 - 0.3
    ) / learning_rate  # Difference of 0.2, normalized by learning rate

    # The average steps behind should be greater than 0 (indicating a difference)
    assert mismatched_result["avg_steps_behind"] > 0

    # For parameters we modified, the difference should be close to our expected value
    # We use a tolerance because the average includes all parameters
    assert mismatched_result["avg_steps_behind"] == pytest.approx(
        expected_diff, abs=0.5
    )


# ---------------------------------------------------------------------- #
#                     NEW TESTS FOR param_avg_change                     #
# ---------------------------------------------------------------------- #


# helper – returns a param_avg_change dict that matches “slice 0-2” length
def _make_avg_change(model: nn.Module, value: float) -> dict[str, torch.Tensor]:
    d: dict[str, torch.Tensor] = {}
    for n, _ in model.named_parameters():
        d[n] = torch.full((2,), value)
    return d


@pytest.mark.asyncio
async def test_avg_change_one_step(setup_model):
    """With param_avg_change == true step size, avg_steps_behind ≃ 1."""
    model = setup_model

    step = 0.05  # custom step size for this test
    param_avg_change = _make_avg_change(model, step)

    debug_dict = {}
    for name, param in model.named_parameters():
        base = param.flatten()[:2].cpu()
        debug_dict[name + "_debug"] = (base + step).tolist()  # exactly one step

    res = await compare_model_with_debug_dict(
        model,
        debug_dict,
        learning_rate=0.01,  # LR should be ignored here
        param_avg_change=param_avg_change,
        index_range=(0, 2),
    )

    assert res["success"] is True
    assert res["avg_steps_behind"] == pytest.approx(1.0, abs=1e-2)
    assert res["max_steps_behind"] == pytest.approx(1.0, abs=1e-2)


@pytest.mark.asyncio
async def test_avg_change_half_step(setup_model):
    """If avg-change is half the true diff, we expect ≃ 2 steps behind."""
    model = setup_model

    true_step = 0.04
    avg_change = true_step / 2  # tell the function updates are smaller

    param_avg_change = _make_avg_change(model, avg_change)

    debug_dict = {}
    for name, param in model.named_parameters():
        base = param.flatten()[:2].cpu()
        debug_dict[name + "_debug"] = (base + true_step).tolist()

    res = await compare_model_with_debug_dict(
        model,
        debug_dict,
        learning_rate=0.01,
        param_avg_change=param_avg_change,
        index_range=(0, 2),
    )

    assert res["avg_steps_behind"] == pytest.approx(2.0, abs=1e-2)


@pytest.mark.asyncio
async def test_avg_change_length_mismatch_fallback(setup_model):
    """
    If the stored slice length is wrong the helper should fall back to LR.
    Expect ≃ 1 step with *learning_rate* instead of the bogus avg-change tensor.
    """
    model = setup_model
    lr = 0.01

    # Build an avg_change dict with *wrong* tensor length (size 1)
    param_avg_change = {n: torch.tensor([lr]) for n, _ in model.named_parameters()}

    debug_dict = {}
    for name, param in model.named_parameters():
        base = param.flatten()[:2].cpu()
        debug_dict[name + "_debug"] = (base + lr).tolist()  # exactly LR ahead

    res = await compare_model_with_debug_dict(
        model,
        debug_dict,
        learning_rate=lr,
        param_avg_change=param_avg_change,
        index_range=(0, 2),
    )

    # Fallback should make it behave like LR-based comparison → 1 step
    assert res["avg_steps_behind"] == pytest.approx(1.0, abs=1e-2)
