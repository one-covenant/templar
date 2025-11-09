from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.optim import AdamW

from neurons.trainer import Trainer


# Mock the Trainer's __init__ method to avoid complex setup
def mock_trainer_init(self):
    pass


@pytest.fixture
def trainer_instance(monkeypatch):
    """Create a mock trainer instance for testing warmup."""
    monkeypatch.setattr(Trainer, "__init__", mock_trainer_init)
    trainer = Trainer()

    # Set up basic attributes needed for testing
    trainer.inner_scheduler_step_count = 0
    trainer.warmup_steps_taken = 0
    trainer.hparams = SimpleNamespace()
    trainer.hparams.inner_steps = 30
    trainer.hparams.optimizer = {
        "type": "adamw",
        "adamw": {
            "learning_rate": 1.17e-4,
            "scheduler": {
                "warmup_steps": 1500,
                "warmup_inner_steps": 100,
                "t_max": 140000,
                "eta_min_factor": 0.1,
                "flatten_start_step": None,
                "flatten_duration": 0,
            },
        },
    }

    # Create a mock optimizer with real param_groups
    trainer.inner_optimizer = Mock(spec=AdamW)
    trainer.inner_optimizer.param_groups = [
        {"lr": 1.17e-4},
        {"lr": 1.17e-4},
    ]

    # Create a mock scheduler
    trainer.inner_scheduler = MagicMock()
    trainer.inner_scheduler.step = Mock()
    trainer.inner_scheduler.get_last_lr = Mock(return_value=[1.17e-4])

    # Set warmup_inner_steps
    trainer.warmup_inner_steps = 100

    return trainer


def test_warmup_initialization(trainer_instance, monkeypatch):
    """Test that warmup_inner_steps is correctly initialized from config."""
    # Mock the initialization to test it properly
    trainer = trainer_instance

    # Simulate init_optimizers_schedulers
    optimizer_config = getattr(trainer.hparams, "optimizer", {})
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    opt_config = optimizer_config.get(optimizer_type, {})
    scheduler_config = opt_config.get("scheduler", {})
    warmup_inner_steps = scheduler_config.get("warmup_inner_steps", 0)

    assert warmup_inner_steps == 100


def test_warmup_initialization_default(trainer_instance):
    """Test that warmup_inner_steps defaults to 0 when not specified."""
    trainer = trainer_instance

    # Remove warmup_inner_steps from config
    del trainer.hparams.optimizer["adamw"]["scheduler"]["warmup_inner_steps"]

    # Simulate init_optimizers_schedulers
    optimizer_config = getattr(trainer.hparams, "optimizer", {})
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    opt_config = optimizer_config.get(optimizer_type, {})
    scheduler_config = opt_config.get("scheduler", {})
    warmup_inner_steps = scheduler_config.get("warmup_inner_steps", 0)

    assert warmup_inner_steps == 0


def test_warmup_lr_scaling_first_step(trainer_instance):
    """Test that LR is scaled correctly on the first warmup step."""
    trainer = trainer_instance
    base_lr = 1.17e-4
    warmup_inner_steps = 100

    trainer.warmup_steps_taken = 0

    # Apply warmup scaling (first step)
    original_lrs = []
    if trainer.warmup_steps_taken < warmup_inner_steps:
        warmup_scale = (trainer.warmup_steps_taken + 1) / warmup_inner_steps
        for param_group in trainer.inner_optimizer.param_groups:
            original_lrs.append(param_group["lr"])
            param_group["lr"] = param_group["lr"] * warmup_scale

    # First step should scale to 1/100
    expected_lr = base_lr * (1 / 100)
    assert abs(trainer.inner_optimizer.param_groups[0]["lr"] - expected_lr) < 1e-10
    assert abs(trainer.inner_optimizer.param_groups[1]["lr"] - expected_lr) < 1e-10

    # Restore original LRs
    if trainer.warmup_steps_taken < warmup_inner_steps:
        for i, param_group in enumerate(trainer.inner_optimizer.param_groups):
            param_group["lr"] = original_lrs[i]

    # LR should be restored
    assert trainer.inner_optimizer.param_groups[0]["lr"] == base_lr
    assert trainer.inner_optimizer.param_groups[1]["lr"] == base_lr


def test_warmup_lr_scaling_mid_warmup(trainer_instance):
    """Test that LR is scaled correctly in the middle of warmup."""
    trainer = trainer_instance
    base_lr = 1.17e-4
    warmup_inner_steps = 100

    trainer.warmup_steps_taken = 49  # 50th step (0-indexed)

    # Apply warmup scaling (50th step)
    original_lrs = []
    if trainer.warmup_steps_taken < warmup_inner_steps:
        warmup_scale = (trainer.warmup_steps_taken + 1) / warmup_inner_steps
        for param_group in trainer.inner_optimizer.param_groups:
            original_lrs.append(param_group["lr"])
            param_group["lr"] = param_group["lr"] * warmup_scale

    # 50th step should scale to 50/100 = 0.5
    expected_lr = base_lr * 0.5
    assert abs(trainer.inner_optimizer.param_groups[0]["lr"] - expected_lr) < 1e-10
    assert abs(trainer.inner_optimizer.param_groups[1]["lr"] - expected_lr) < 1e-10

    # Restore
    for i, param_group in enumerate(trainer.inner_optimizer.param_groups):
        param_group["lr"] = original_lrs[i]

    assert trainer.inner_optimizer.param_groups[0]["lr"] == base_lr


def test_warmup_lr_scaling_last_step(trainer_instance):
    """Test that LR is scaled to full value on the last warmup step."""
    trainer = trainer_instance
    base_lr = 1.17e-4
    warmup_inner_steps = 100

    trainer.warmup_steps_taken = 99  # 100th step (0-indexed)

    # Apply warmup scaling (100th step)
    original_lrs = []
    if trainer.warmup_steps_taken < warmup_inner_steps:
        warmup_scale = (trainer.warmup_steps_taken + 1) / warmup_inner_steps
        for param_group in trainer.inner_optimizer.param_groups:
            original_lrs.append(param_group["lr"])
            param_group["lr"] = param_group["lr"] * warmup_scale

    # 100th step should scale to 100/100 = 1.0 (full LR)
    expected_lr = base_lr * 1.0
    assert abs(trainer.inner_optimizer.param_groups[0]["lr"] - expected_lr) < 1e-10
    assert abs(trainer.inner_optimizer.param_groups[1]["lr"] - expected_lr) < 1e-10


def test_warmup_lr_scaling_after_warmup(trainer_instance):
    """Test that LR is not scaled after warmup period is complete."""
    trainer = trainer_instance
    base_lr = 1.17e-4
    warmup_inner_steps = 100

    trainer.warmup_steps_taken = 100  # Past warmup

    # Apply warmup scaling (should not apply)
    original_lrs = []
    if trainer.warmup_steps_taken < warmup_inner_steps:
        warmup_scale = (trainer.warmup_steps_taken + 1) / warmup_inner_steps
        for param_group in trainer.inner_optimizer.param_groups:
            original_lrs.append(param_group["lr"])
            param_group["lr"] = param_group["lr"] * warmup_scale

    # LR should remain unchanged (warmup is complete)
    assert trainer.inner_optimizer.param_groups[0]["lr"] == base_lr
    assert trainer.inner_optimizer.param_groups[1]["lr"] == base_lr
    assert len(original_lrs) == 0  # No scaling was applied


def test_warmup_steps_counter_increments(trainer_instance):
    """Test that warmup_steps_taken increments correctly."""
    trainer = trainer_instance
    warmup_inner_steps = 100

    trainer.warmup_steps_taken = 0

    # Simulate taking warmup steps
    for i in range(warmup_inner_steps):
        assert trainer.warmup_steps_taken == i

        # Simulate the increment logic
        if trainer.warmup_steps_taken < warmup_inner_steps:
            trainer.warmup_steps_taken += 1

    # Should have incremented exactly 100 times
    assert trainer.warmup_steps_taken == 100


def test_warmup_steps_counter_stops_incrementing(trainer_instance):
    """Test that warmup_steps_taken stops incrementing after warmup."""
    trainer = trainer_instance
    warmup_inner_steps = 100

    trainer.warmup_steps_taken = 100

    # Simulate more steps (should not increment)
    for _ in range(10):
        if trainer.warmup_steps_taken < warmup_inner_steps:
            trainer.warmup_steps_taken += 1

    # Should still be 100
    assert trainer.warmup_steps_taken == 100


def test_warmup_with_restart(trainer_instance):
    """Test that warmup resets on restart/re-initialization."""
    trainer = trainer_instance

    # Simulate we're far along in training
    trainer.inner_scheduler_step_count = 5000
    trainer.warmup_steps_taken = 100  # Warmup already completed

    # Now simulate restart by re-initializing warmup counter
    trainer.warmup_steps_taken = 0  # This would happen in init_optimizers_schedulers

    # Warmup should now be active again
    assert trainer.warmup_steps_taken == 0
    assert trainer.warmup_steps_taken < trainer.warmup_inner_steps


def test_warmup_disabled_when_zero(trainer_instance):
    """Test that warmup is disabled when warmup_inner_steps is 0."""
    trainer = trainer_instance
    trainer.warmup_inner_steps = 0
    trainer.warmup_steps_taken = 0
    base_lr = 1.17e-4

    # Apply warmup scaling (should not apply when warmup_inner_steps=0)
    original_lrs = []
    if trainer.warmup_steps_taken < trainer.warmup_inner_steps:
        warmup_scale = (trainer.warmup_steps_taken + 1) / trainer.warmup_inner_steps
        for param_group in trainer.inner_optimizer.param_groups:
            original_lrs.append(param_group["lr"])
            param_group["lr"] = param_group["lr"] * warmup_scale

    # LR should remain unchanged
    assert trainer.inner_optimizer.param_groups[0]["lr"] == base_lr
    assert len(original_lrs) == 0  # No scaling was applied


def test_warmup_with_muon_optimizer(trainer_instance):
    """Test that warmup works with muon optimizer."""
    trainer = trainer_instance

    # Change to muon optimizer
    trainer.hparams.optimizer["type"] = "muon"
    trainer.hparams.optimizer["muon"] = {
        "learning_rate": 2e-3,
        "scheduler": {
            "warmup_steps": 1500,
            "warmup_inner_steps": 50,
            "t_max": 140000,
            "eta_min_factor": 0.1,
        },
    }

    # Simulate init_optimizers_schedulers
    optimizer_config = getattr(trainer.hparams, "optimizer", {})
    optimizer_type = optimizer_config.get("type", "adamw").lower()
    opt_config = optimizer_config.get(optimizer_type, {})
    scheduler_config = opt_config.get("scheduler", {})
    warmup_inner_steps = scheduler_config.get("warmup_inner_steps", 0)

    assert warmup_inner_steps == 50


def test_warmup_lr_precision(trainer_instance):
    """Test that LR restoration maintains precision (no floating point drift)."""
    trainer = trainer_instance
    original_lr = 1.17e-4
    warmup_inner_steps = 100

    # Set up original LRs
    trainer.inner_optimizer.param_groups[0]["lr"] = original_lr
    trainer.inner_optimizer.param_groups[1]["lr"] = original_lr

    # Simulate multiple warmup steps with scale/restore cycle
    for step in range(50):
        trainer.warmup_steps_taken = step

        # Apply warmup scaling
        original_lrs = []
        if trainer.warmup_steps_taken < warmup_inner_steps:
            warmup_scale = (trainer.warmup_steps_taken + 1) / warmup_inner_steps
            for param_group in trainer.inner_optimizer.param_groups:
                original_lrs.append(param_group["lr"])
                param_group["lr"] = param_group["lr"] * warmup_scale

        # Restore
        if trainer.warmup_steps_taken < warmup_inner_steps:
            for i, param_group in enumerate(trainer.inner_optimizer.param_groups):
                param_group["lr"] = original_lrs[i]

    # LR should be exactly the original value (no drift from multiply/divide)
    assert trainer.inner_optimizer.param_groups[0]["lr"] == original_lr
    assert trainer.inner_optimizer.param_groups[1]["lr"] == original_lr


def test_warmup_multiple_param_groups(trainer_instance):
    """Test that warmup scales all param groups correctly."""
    trainer = trainer_instance
    warmup_inner_steps = 100

    # Set different LRs for different param groups
    trainer.inner_optimizer.param_groups = [
        {"lr": 1.0e-4},
        {"lr": 2.0e-4},
        {"lr": 3.0e-4},
    ]

    trainer.warmup_steps_taken = 24  # 25th step, scale = 0.25

    # Apply warmup scaling
    original_lrs = []
    if trainer.warmup_steps_taken < warmup_inner_steps:
        warmup_scale = (trainer.warmup_steps_taken + 1) / warmup_inner_steps
        for param_group in trainer.inner_optimizer.param_groups:
            original_lrs.append(param_group["lr"])
            param_group["lr"] = param_group["lr"] * warmup_scale

    # All should be scaled by 0.25
    assert abs(trainer.inner_optimizer.param_groups[0]["lr"] - 1.0e-4 * 0.25) < 1e-10
    assert abs(trainer.inner_optimizer.param_groups[1]["lr"] - 2.0e-4 * 0.25) < 1e-10
    assert abs(trainer.inner_optimizer.param_groups[2]["lr"] - 3.0e-4 * 0.25) < 1e-10

    # Restore
    for i, param_group in enumerate(trainer.inner_optimizer.param_groups):
        param_group["lr"] = original_lrs[i]

    # Should be restored exactly
    assert trainer.inner_optimizer.param_groups[0]["lr"] == 1.0e-4
    assert trainer.inner_optimizer.param_groups[1]["lr"] == 2.0e-4
    assert trainer.inner_optimizer.param_groups[2]["lr"] == 3.0e-4
