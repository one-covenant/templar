#!/usr/bin/env python3
"""
Test suite for consecutive negative evaluation exclusion logic in the validator.
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class MockValidator:
    """Mock validator with simplified exclusion logic for testing."""

    def __init__(self, consecutive_negative_threshold=3, exclude_negative_peers=True):
        self.hparams = SimpleNamespace(
            consecutive_negative_threshold=consecutive_negative_threshold,
            exclude_negative_peers=exclude_negative_peers,
        )
        self.peer_eval_history = {}
        self.consecutive_negative_count = {}
        self.excluded_from_gather = set()
        self.exclusion_start_window = {}
        self.gradient_scores = {}
        self.sync_window = 0
        self.current_window = 0
        self.eval_history_limit = 20
        self.global_step = 0
        self.wandb = MagicMock()

    def should_exclude_from_gather(self, uid):
        """Check if a peer should be excluded from gather selection."""
        if not self.hparams.exclude_negative_peers:
            return False

        threshold = self.hparams.consecutive_negative_threshold
        consecutive_negatives = self.consecutive_negative_count.get(uid, 0)

        return consecutive_negatives >= threshold

    def track_negative_evaluation(self, eval_uid, is_negative):
        """Simplified tracking for testing."""
        # Initialize if needed
        if eval_uid not in self.peer_eval_history:
            self.peer_eval_history[eval_uid] = deque(maxlen=self.eval_history_limit)
            self.consecutive_negative_count[eval_uid] = 0

        window = self.peer_eval_history[eval_uid]
        window.append(is_negative)

        # Update consecutive negative count
        prev_consecutive = self.consecutive_negative_count.get(eval_uid, 0)
        if is_negative:
            self.consecutive_negative_count[eval_uid] = prev_consecutive + 1

            # Check if should exclude
            if (
                self.hparams.exclude_negative_peers
                and self.consecutive_negative_count[eval_uid]
                >= self.hparams.consecutive_negative_threshold
                and eval_uid not in self.excluded_from_gather
            ):
                self.excluded_from_gather.add(eval_uid)
                self.exclusion_start_window[eval_uid] = self.sync_window
        else:
            # Recovery mechanism - reset on positive evaluation
            if eval_uid in self.excluded_from_gather:
                self.excluded_from_gather.discard(eval_uid)
                self.exclusion_start_window.pop(eval_uid, None)

            self.consecutive_negative_count[eval_uid] = 0


@pytest.fixture
def validator():
    """Create a mock validator instance for testing."""
    return MockValidator(consecutive_negative_threshold=3, exclude_negative_peers=True)


class TestConsecutiveNegativeExclusion:
    """Test cases for consecutive negative evaluation exclusion."""

    def test_exclusion_after_threshold(self, validator):
        """Test that a peer is excluded after reaching the consecutive negative threshold."""
        uid = 100

        # First negative evaluation
        validator.sync_window = 1
        validator.track_negative_evaluation(uid, True)
        assert validator.consecutive_negative_count[uid] == 1
        assert not validator.should_exclude_from_gather(uid)

        # Second negative evaluation
        validator.sync_window = 2
        validator.track_negative_evaluation(uid, True)
        assert validator.consecutive_negative_count[uid] == 2
        assert not validator.should_exclude_from_gather(uid)

        # Third negative evaluation - should trigger exclusion
        validator.sync_window = 3
        validator.track_negative_evaluation(uid, True)
        assert validator.consecutive_negative_count[uid] == 3
        assert validator.should_exclude_from_gather(uid)
        assert uid in validator.excluded_from_gather

    def test_recovery_with_positive_evaluation(self, validator):
        """Test that a peer recovers and is re-included after a positive evaluation."""
        uid = 100

        # Get peer excluded first
        for window in range(1, 4):
            validator.sync_window = window
            validator.track_negative_evaluation(uid, True)

        assert validator.should_exclude_from_gather(uid)
        assert uid in validator.excluded_from_gather

        # Positive evaluation should trigger recovery
        validator.sync_window = 4
        validator.track_negative_evaluation(uid, False)

        assert validator.consecutive_negative_count[uid] == 0
        assert not validator.should_exclude_from_gather(uid)
        assert uid not in validator.excluded_from_gather

    def test_mixed_evaluations_no_exclusion(self, validator):
        """Test that mixed evaluations don't trigger exclusion."""
        uid = 200

        # Two negative evaluations
        validator.sync_window = 1
        validator.track_negative_evaluation(uid, True)
        validator.sync_window = 2
        validator.track_negative_evaluation(uid, True)
        assert validator.consecutive_negative_count[uid] == 2

        # Positive evaluation resets the count
        validator.sync_window = 3
        validator.track_negative_evaluation(uid, False)
        assert validator.consecutive_negative_count[uid] == 0

        # Another negative starts fresh
        validator.sync_window = 4
        validator.track_negative_evaluation(uid, True)
        assert validator.consecutive_negative_count[uid] == 1
        assert not validator.should_exclude_from_gather(uid)

    def test_feature_disabled(self):
        """Test that exclusion doesn't occur when feature is disabled."""
        validator = MockValidator(
            consecutive_negative_threshold=3, exclude_negative_peers=False
        )
        uid = 300

        # Add 5 consecutive negative evaluations
        for window in range(1, 6):
            validator.sync_window = window
            validator.track_negative_evaluation(uid, True)

        # Should not be excluded even with 5 consecutive negatives
        assert validator.consecutive_negative_count[uid] == 5
        assert not validator.should_exclude_from_gather(uid)
        assert uid not in validator.excluded_from_gather

    def test_different_thresholds(self):
        """Test exclusion with different threshold values."""
        # Test with threshold of 2
        validator = MockValidator(
            consecutive_negative_threshold=2, exclude_negative_peers=True
        )
        uid = 400

        validator.sync_window = 1
        validator.track_negative_evaluation(uid, True)
        assert not validator.should_exclude_from_gather(uid)

        validator.sync_window = 2
        validator.track_negative_evaluation(uid, True)
        assert validator.should_exclude_from_gather(uid)

        # Test with threshold of 5
        validator = MockValidator(
            consecutive_negative_threshold=5, exclude_negative_peers=True
        )
        uid = 500

        for window in range(1, 5):
            validator.sync_window = window
            validator.track_negative_evaluation(uid, True)
            assert not validator.should_exclude_from_gather(uid)

        validator.sync_window = 5
        validator.track_negative_evaluation(uid, True)
        assert validator.should_exclude_from_gather(uid)

    def test_multiple_peers_independent_tracking(self, validator):
        """Test that multiple peers are tracked independently."""
        uid1, uid2, uid3 = 100, 200, 300

        # UID1: Gets excluded
        for window in range(1, 4):
            validator.sync_window = window
            validator.track_negative_evaluation(uid1, True)

        # UID2: Gets 2 negatives then recovers
        validator.sync_window = 1
        validator.track_negative_evaluation(uid2, True)
        validator.sync_window = 2
        validator.track_negative_evaluation(uid2, True)
        validator.sync_window = 3
        validator.track_negative_evaluation(uid2, False)

        # UID3: All positive
        for window in range(1, 4):
            validator.sync_window = window
            validator.track_negative_evaluation(uid3, False)

        assert validator.should_exclude_from_gather(uid1)  # Excluded
        assert not validator.should_exclude_from_gather(uid2)  # Recovered
        assert not validator.should_exclude_from_gather(uid3)  # Never negative

    def test_exclusion_window_tracking(self, validator):
        """Test that exclusion start window is properly tracked."""
        uid = 100

        # Get peer excluded at window 5
        for window in range(3, 6):
            validator.sync_window = window
            validator.track_negative_evaluation(uid, True)

        assert uid in validator.excluded_from_gather
        assert validator.exclusion_start_window[uid] == 5  # Excluded at window 5

        # Recover at window 10
        validator.sync_window = 10
        validator.track_negative_evaluation(uid, False)

        assert uid not in validator.excluded_from_gather
        assert uid not in validator.exclusion_start_window

    def test_evaluation_history_maintained(self, validator):
        """Test that evaluation history is properly maintained."""
        uid = 100

        # Add some evaluations
        evaluations = [True, True, False, True, False, True, True, True]
        for i, is_negative in enumerate(evaluations):
            validator.sync_window = i + 1
            validator.track_negative_evaluation(uid, is_negative)

        # Check history is maintained
        assert len(validator.peer_eval_history[uid]) == len(evaluations)
        assert list(validator.peer_eval_history[uid]) == evaluations

        # Last three are True, so should be excluded
        assert validator.consecutive_negative_count[uid] == 3
        assert validator.should_exclude_from_gather(uid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
