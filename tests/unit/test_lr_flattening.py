from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from neurons.trainer import Trainer


# Mock the Trainer's __init__ method to avoid complex setup
def mock_trainer_init(self):
    pass


@pytest.fixture
def trainer_instance(monkeypatch):
    """Create a mock trainer instance for testing."""
    monkeypatch.setattr(Trainer, "__init__", mock_trainer_init)
    trainer = Trainer()

    # Set up basic attributes needed for testing
    trainer.inner_scheduler_step_count = 0
    trainer.hparams = SimpleNamespace()
    trainer.hparams.inner_steps = 30
    trainer.hparams.optimizer = {
        "type": "adamw",
        "adamw": {
            "learning_rate": 1.17e-4,
            "scheduler": {
                "warmup_steps": 1500,
                "t_max": 140000,
                "eta_min_factor": 0.1,
                "flatten_start_step": None,
                "flatten_duration": 0,
            },
        },
    }

    # Create a mock optimizer and scheduler
    trainer.inner_optimizer = Mock()
    trainer.inner_scheduler = MagicMock()
    trainer.inner_scheduler.step = Mock()
    trainer.inner_scheduler.get_last_lr = Mock(return_value=[1.17e-4])

    return trainer


def test_should_skip_scheduler_step_disabled(trainer_instance):
    """Test that flattening is disabled by default (flatten_start_step is None)."""
    trainer = trainer_instance
    trainer.inner_scheduler_step_count = 3000  # Window 100

    # flatten_start_step is None by default
    assert trainer.should_skip_scheduler_step() is False


def test_should_skip_scheduler_step_zero_duration(trainer_instance):
    """Test that flattening is disabled when flatten_duration is 0."""
    trainer = trainer_instance
    trainer.inner_scheduler_step_count = 3000  # Window 100

    # Set flatten_start_step but keep duration at 0
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 0

    assert trainer.should_skip_scheduler_step() is False


def test_should_skip_scheduler_step_before_flatten(trainer_instance):
    """Test that flattening is not active before the flatten window."""
    trainer = trainer_instance

    # Flatten from window 100 for 10 windows (steps 3000-3299)
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 10

    # Before flatten window
    trainer.inner_scheduler_step_count = 2999
    assert trainer.should_skip_scheduler_step() is False


def test_should_skip_scheduler_step_during_flatten(trainer_instance):
    """Test that flattening is active during the flatten window."""
    trainer = trainer_instance

    # Flatten from window 100 for 10 windows (steps 3000-3299)
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 10

    # At start of flatten window
    trainer.inner_scheduler_step_count = 3000
    assert trainer.should_skip_scheduler_step() is True

    # In middle of flatten window
    trainer.inner_scheduler_step_count = 3150
    assert trainer.should_skip_scheduler_step() is True

    # At end of flatten window (last step that should be flattened)
    trainer.inner_scheduler_step_count = 3299
    assert trainer.should_skip_scheduler_step() is True


def test_should_skip_scheduler_step_after_flatten(trainer_instance):
    """Test that flattening is not active after the flatten window."""
    trainer = trainer_instance

    # Flatten from window 100 for 10 windows (steps 3000-3299)
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 10

    # After flatten window
    trainer.inner_scheduler_step_count = 3300
    assert trainer.should_skip_scheduler_step() is False

    trainer.inner_scheduler_step_count = 5000
    assert trainer.should_skip_scheduler_step() is False


def test_should_skip_scheduler_step_with_muon(trainer_instance):
    """Test that flattening works with muon optimizer."""
    trainer = trainer_instance

    # Change to muon optimizer
    trainer.hparams.optimizer["type"] = "muon"
    trainer.hparams.optimizer["muon"] = {
        "learning_rate": 2e-3,
        "scheduler": {
            "warmup_steps": 1500,
            "t_max": 140000,
            "eta_min_factor": 0.1,
            "flatten_start_step": 50,
            "flatten_duration": 5,
        },
    }

    # Before flatten (window 50 = steps 1500-1649)
    trainer.inner_scheduler_step_count = 1499
    assert trainer.should_skip_scheduler_step() is False

    # During flatten
    trainer.inner_scheduler_step_count = 1500
    assert trainer.should_skip_scheduler_step() is True

    trainer.inner_scheduler_step_count = 1600
    assert trainer.should_skip_scheduler_step() is True

    # After flatten (window 55 = step 1650+)
    trainer.inner_scheduler_step_count = 1650
    assert trainer.should_skip_scheduler_step() is False


def test_should_skip_scheduler_step_conversion(trainer_instance):
    """Test the outer step to inner step conversion is correct."""
    trainer = trainer_instance

    # Flatten from window 200 for 20 windows
    # With inner_steps=30: steps 6000-6599 should be flattened
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 200
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 20

    # Just before window 200 (step 5999)
    trainer.inner_scheduler_step_count = 5999
    assert trainer.should_skip_scheduler_step() is False

    # First step of window 200 (step 6000)
    trainer.inner_scheduler_step_count = 6000
    assert trainer.should_skip_scheduler_step() is True

    # Last step of window 219 (step 6599)
    trainer.inner_scheduler_step_count = 6599
    assert trainer.should_skip_scheduler_step() is True

    # First step of window 220 (step 6600)
    trainer.inner_scheduler_step_count = 6600
    assert trainer.should_skip_scheduler_step() is False


def test_scheduler_not_stepped_during_flatten(trainer_instance):
    """Test that scheduler.step() is not called during flatten window."""
    trainer = trainer_instance

    # Enable flattening
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 10

    # Simulate training steps during flatten window
    trainer.inner_scheduler_step_count = 3000

    # Reset mock
    trainer.inner_scheduler.step.reset_mock()

    # Simulate what happens in the training loop
    for _ in range(30):  # One window worth of inner steps
        if not trainer.should_skip_scheduler_step():
            trainer.inner_scheduler.step()
        trainer.inner_scheduler_step_count += 1

    # Scheduler should NOT have been stepped at all (we're in flatten window)
    assert trainer.inner_scheduler.step.call_count == 0
    assert trainer.inner_scheduler_step_count == 3030


def test_scheduler_stepped_after_flatten(trainer_instance):
    """Test that scheduler.step() is called after flatten window."""
    trainer = trainer_instance

    # Enable flattening
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 10

    # Start after flatten window
    trainer.inner_scheduler_step_count = 3300

    # Reset mock
    trainer.inner_scheduler.step.reset_mock()

    # Simulate what happens in the training loop
    for _ in range(30):  # One window worth of inner steps
        if not trainer.should_skip_scheduler_step():
            trainer.inner_scheduler.step()
        trainer.inner_scheduler_step_count += 1

    # Scheduler should have been stepped every time (we're past flatten window)
    assert trainer.inner_scheduler.step.call_count == 30
    assert trainer.inner_scheduler_step_count == 3330


def test_scheduler_stepped_before_flatten(trainer_instance):
    """Test that scheduler.step() is called before flatten window."""
    trainer = trainer_instance

    # Enable flattening
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 10

    # Start before flatten window
    trainer.inner_scheduler_step_count = 2970

    # Reset mock
    trainer.inner_scheduler.step.reset_mock()

    # Simulate what happens in the training loop
    for _ in range(30):  # One window worth of inner steps
        if not trainer.should_skip_scheduler_step():
            trainer.inner_scheduler.step()
        trainer.inner_scheduler_step_count += 1

    # Scheduler should have been stepped every time (we're before flatten window)
    assert trainer.inner_scheduler.step.call_count == 30
    assert trainer.inner_scheduler_step_count == 3000


def test_partial_window_flatten_transition(trainer_instance):
    """Test behavior when a window partially overlaps with flatten start."""
    trainer = trainer_instance

    # Flatten from window 100 for 10 windows (steps 3000-3299)
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_start_step"] = 100
    trainer.hparams.optimizer["adamw"]["scheduler"]["flatten_duration"] = 10

    # Start 10 steps before flatten window (step 2990)
    trainer.inner_scheduler_step_count = 2990

    # Reset mock
    trainer.inner_scheduler.step.reset_mock()

    # Simulate a full window of 30 steps
    steps_before_flatten = 0
    steps_during_flatten = 0

    for _ in range(30):
        if not trainer.should_skip_scheduler_step():
            trainer.inner_scheduler.step()
            steps_before_flatten += 1
        else:
            steps_during_flatten += 1
        trainer.inner_scheduler_step_count += 1

    # Should have stepped 10 times (steps 2990-2999) then stopped (steps 3000-3019)
    assert trainer.inner_scheduler.step.call_count == 10
    assert steps_before_flatten == 10
    assert steps_during_flatten == 20
    assert trainer.inner_scheduler_step_count == 3020


def test_inner_scheduler_step_count_persists(trainer_instance):
    """Test that inner_scheduler_step_count is correctly tracked."""
    trainer = trainer_instance

    # No flattening for this test
    trainer.inner_scheduler_step_count = 0

    # Simulate multiple windows
    for window in range(5):
        for _ in range(30):  # inner_steps per window
            if not trainer.should_skip_scheduler_step():
                trainer.inner_scheduler.step()
            trainer.inner_scheduler_step_count += 1

    # After 5 windows of 30 steps each
    assert trainer.inner_scheduler_step_count == 150
