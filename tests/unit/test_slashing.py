from collections import deque
from types import SimpleNamespace

import pytest
import torch

from neurons.validator import Validator
from tplr.neurons import determine_slash_egregiousness, instantiate_slashing_multiplier


# Mock the Validator's __init__ method to avoid complex setup
def mock_validator_init(self):
    pass


@pytest.fixture
def validator_instance(monkeypatch):
    """Test validator instance"""
    monkeypatch.setattr(Validator, "__init__", mock_validator_init)
    validator = Validator()
    validator.final_scores = torch.ones(10, dtype=torch.float32)
    validator.weights = torch.zeros(10, dtype=torch.float32)
    validator.gradient_scores = torch.zeros(10, dtype=torch.float32)
    validator.binary_moving_averages = torch.ones(10, dtype=torch.float32)
    validator.binary_indicator_scores = torch.zeros(10, dtype=torch.float32)
    validator.sync_scores = torch.zeros(10, dtype=torch.float32)
    validator.openskill_ratings = {}
    validator.eval_peers = {}
    validator.inactive_scores = {}
    validator.idx_similarity_slashing_rate = instantiate_slashing_multiplier()
    validator.naughty_peers = {}
    validator.naughty_peer_timeout = 200
    validator.sync_window = 0
    validator.current_window = 0
    validator.peer_eval_history = {}

    # Add new attributes for consecutive negative tracking
    validator.consecutive_negative_count = {}
    validator.consecutive_missing_gradient_count = {}
    validator.missing_gradient_history = {}
    validator.missing_gradient_history_limit = 8
    validator.excluded_from_gather = set()
    validator.exclusion_start_window = {}
    validator.eval_history_limit = 20
    validator.score_zero_threshold = 1e-4
    validator.evaluated_uids = set()
    validator.peers_last_eval_window = {}
    validator.global_step = 0
    validator.missing_gradient_penalty_score = -99.0

    # Mock wandb
    validator.wandb = SimpleNamespace()
    validator.wandb.log = lambda *args, **kwargs: None

    # Mock methods
    validator.record_missing_gradient_for_openskill = lambda uid: None

    # Mock comms and metagraph
    validator.comms = SimpleNamespace()
    validator.comms.metagraph = SimpleNamespace()
    validator.comms.metagraph.uids = [0, 1, 2, 3, 4, 5, 6]
    validator.comms.metagraph.hotkeys = [
        "hotkey_0",
        "hotkey_1",
        "hotkey_2",
        "hotkey_3",
        "hotkey_4",
        "hotkey_5",
        "hotkey_6_new",
    ]
    validator.current_hotkeys = {
        0: "hotkey_0",
        1: "hotkey_1",
        2: "hotkey_2",
        3: "hotkey_3",
        4: "hotkey_4",
        5: "hotkey_5",
        6: "hotkey_6_old",
    }

    return validator


def test_determine_slash_egregiousness():
    # 0.0 won't make it to this function for miner clarity
    assert determine_slash_egregiousness(0.0) == "high"

    # high/max threshold
    assert determine_slash_egregiousness(0.499) == "high"
    assert determine_slash_egregiousness(0.501) == "max"

    # max/mega threshold
    assert determine_slash_egregiousness(0.599) == "max"
    assert determine_slash_egregiousness(0.601) == "mega"
    assert determine_slash_egregiousness(1.0) == "mega"

    # Test invalid inputs if validation is added
    with pytest.raises(ValueError):
        determine_slash_egregiousness(-0.1)
    with pytest.raises(ValueError):
        determine_slash_egregiousness(1.1)


def test_instantiate_slashing_multiplier():
    output_dict = instantiate_slashing_multiplier()
    assert isinstance(output_dict, dict)
    assert all(isinstance(val, float) for val in output_dict.values())
    assert all(isinstance(key, str) for key in output_dict)


def test_slash_from_overlap(validator_instance):
    validator = validator_instance

    # Test case 1: High overlap
    idx_overlap_high = {"uids_over_thresh": {1: "high"}}
    validator.slash_from_overlap(idx_overlap_high)
    assert validator.final_scores[1] == 0.5
    assert validator.binary_moving_averages[1] == 0.5

    # Test case 2: Max overlap
    idx_overlap_max = {"uids_over_thresh": {2: "max"}}
    validator.final_scores[2] = 1.0  # reset score
    validator.binary_moving_averages[2] = 1.0  # reset score
    validator.inactive_scores[2] = (0, 1.0)  # Ensure peer exists in inactive_scores
    validator.slash_from_overlap(idx_overlap_max)
    assert validator.final_scores[2] == 0.0
    assert validator.binary_moving_averages[2] == 0.0

    # Test case 3: Mega overlap
    idx_overlap_mega = {"uids_over_thresh": {3: "mega"}}
    validator.final_scores[3] = 1.0  # reset score
    validator.binary_moving_averages[3] = 1.0  # reset score
    validator.inactive_scores[3] = (0, 1.0)  # Ensure peer exists in inactive_scores
    validator.slash_from_overlap(idx_overlap_mega)
    assert validator.final_scores[3] == 0.0
    assert validator.binary_moving_averages[3] == 0.0
    assert 3 in validator.naughty_peers
    assert validator.naughty_peers[3] == validator.naughty_peer_timeout - 1

    # Test case 4: Naughty peer timeout
    validator.naughty_peers = {4: 1}
    validator.inactive_scores[4] = (0, 1.0)  # Ensure peer exists in inactive_scores
    idx_overlap_empty = {"uids_over_thresh": {}}
    validator.slash_from_overlap(idx_overlap_empty)
    assert 4 not in validator.naughty_peers

    # Test case 5: No overlap
    validator.final_scores[5] = 1.0  # reset score
    validator.binary_moving_averages[5] = 1.0  # reset score
    idx_overlap_none = {"uids_over_thresh": {}}
    validator.slash_from_overlap(idx_overlap_none)
    assert validator.final_scores[5] == 1.0
    assert validator.binary_moving_averages[5] == 1.0


def test_check_deregistered_uids(validator_instance):
    validator = validator_instance
    original_hotkeys = validator.current_hotkeys.copy()

    # Test case 1: UID 6's hotkey changed (removed), UID 2's did not (remains)
    validator.current_hotkeys = original_hotkeys.copy()
    idx_overlap_peers = {6: "high", 2: "max"}
    updated_peers = validator.check_deregistered_uids(idx_overlap_peers)
    assert 6 not in updated_peers
    assert 2 in updated_peers

    # Test case 2: UID 2's hotkey did not change, should remain
    validator.current_hotkeys = original_hotkeys.copy()
    idx_overlap_peers = {2: "max"}
    updated_peers = validator.check_deregistered_uids(idx_overlap_peers)
    assert 2 in updated_peers

    # Test case 3: UID 6 is 'mega' and should not be removed even if hotkey changed
    validator.current_hotkeys = original_hotkeys.copy()
    idx_overlap_peers = {6: "mega"}
    validator.naughty_peers = {6: 100}
    updated_peers = validator.check_deregistered_uids(idx_overlap_peers)
    assert 6 in updated_peers

    # Test case 4: UID 6 is in naughty_peers and should be removed (since it's not 'mega')
    validator.current_hotkeys = original_hotkeys.copy()
    validator.naughty_peers = {6: 100}
    idx_overlap_peers = {6: "high"}
    updated_peers = validator.check_deregistered_uids(idx_overlap_peers)
    assert 6 not in updated_peers
    assert 6 not in validator.naughty_peers


def test_slash_for_missing_gradients_escalating(validator_instance):
    """Test escalating penalties for missing gradients: 0.75, 0.5, 0"""
    validator = validator_instance
    validator.hparams = SimpleNamespace()
    validator.hparams.gather_peers_slash_threshold = 0.4

    uid = 1
    success_rate = 0.5  # Above threshold

    # First miss: should multiply by 0.75
    validator.slash_for_missing_gradients([uid], success_rate)
    assert validator.final_scores[uid] == pytest.approx(0.75, rel=1e-3)
    assert validator.consecutive_missing_gradient_count[uid] == 1

    # Second miss: should multiply by 0.5
    validator.slash_for_missing_gradients([uid], success_rate)
    assert validator.final_scores[uid] == pytest.approx(0.75 * 0.5, rel=1e-3)
    assert validator.consecutive_missing_gradient_count[uid] == 2

    # Third miss: should multiply by 0
    validator.slash_for_missing_gradients([uid], success_rate)
    assert validator.final_scores[uid] == 0.0
    assert validator.consecutive_missing_gradient_count[uid] == 3


def test_slash_for_missing_gradients_mega_slash(validator_instance):
    """Test mega slash when >50% of last 8 windows have missing gradients"""
    validator = validator_instance
    validator.hparams = SimpleNamespace()
    validator.hparams.gather_peers_slash_threshold = 0.4

    uid = 2
    success_rate = 0.5

    # Simulate 5 missing and 3 successful gradients (5/8 = 62.5% > 50%)
    validator.missing_gradient_history[uid] = deque(
        [True, False, True, True, False, True, False, True], maxlen=8
    )
    validator.consecutive_missing_gradient_count[uid] = 1

    # Next miss should trigger mega slash
    validator.slash_for_missing_gradients([uid], success_rate)

    # Should be in naughty list
    assert uid in validator.naughty_peers
    assert validator.naughty_peers[uid] == validator.naughty_peer_timeout

    # Should be reset (score should be 0)
    assert validator.final_scores[uid] == 0.0


def test_slash_for_missing_gradients_below_threshold(validator_instance):
    """Test that peers aren't mega slashed when below 50% threshold"""
    validator = validator_instance
    validator.hparams = SimpleNamespace()
    validator.hparams.gather_peers_slash_threshold = 0.4

    uid = 3
    success_rate = 0.5

    # Simulate 2 missing and 5 successful gradients (2/7 initially)
    # After appending one more miss, becomes 3/8 = 37.5% < 50%
    validator.missing_gradient_history[uid] = deque(
        [True, False, False, False, True, False, False], maxlen=8
    )
    validator.consecutive_missing_gradient_count[uid] = 1

    # Next miss should NOT trigger mega slash (3/8 = 37.5% < 50%)
    validator.slash_for_missing_gradients([uid], success_rate)

    # Should NOT be in naughty list (mega slash not triggered)
    assert uid not in validator.naughty_peers

    # Should have normal slash applied (0.5 for second consecutive miss)
    assert validator.final_scores[uid] == pytest.approx(0.5, rel=1e-3)


def test_track_negative_evaluation_slashing(validator_instance):
    """Test slashing when >50% of last 8 evaluations are negative"""
    validator = validator_instance
    validator.hparams = SimpleNamespace()

    uid = 4
    validator.final_scores[uid] = 1.0

    # Simulate 5 negative and 3 positive evaluations in last 8
    # (5/8 = 62.5% > 50%)
    validator.peer_eval_history[uid] = deque(
        [True, False, True, True, False, True, False, True],
        maxlen=validator.eval_history_limit,
    )
    validator.gradient_scores[uid] = -0.1  # Set current evaluation to negative

    # Track the negative evaluation - should trigger slashing
    validator.track_negative_evaluation(uid)

    # Should be slashed by 75% (multiplied by 0.25)
    # But the history already had 5 negatives, now we added one more (6 total)
    # Actually, track_negative_evaluation appends the current evaluation
    # So now we have 6 negatives out of 9 total
    # But we only check last 8, which would be the last 8 in the deque
    # Let me recalculate: after append, we have [True, False, True, True, False, True, False, True, True]
    # Last 8 would be [False, True, True, False, True, False, True, True] = 5/8 = 62.5%
    # So it should trigger slashing
    assert validator.final_scores[uid] < 1.0
    assert validator.final_scores[uid] == pytest.approx(0.25, rel=1e-3)


def test_track_negative_evaluation_no_slash_below_threshold(validator_instance):
    """Test that peers aren't slashed when below 50% negative threshold"""
    validator = validator_instance
    validator.hparams = SimpleNamespace()

    uid = 5
    validator.final_scores[uid] = 1.0

    # Simulate 4 negative and 4 positive evaluations in last 8 (4/8 = 50%, not > 50%)
    validator.peer_eval_history[uid] = deque(
        [True, False, True, False, True, False, True, False],
        maxlen=validator.eval_history_limit,
    )
    validator.gradient_scores[uid] = 0.1  # Set current evaluation to positive

    # Track the positive evaluation - should NOT trigger slashing
    validator.track_negative_evaluation(uid)

    # Score should remain 1.0 (no slashing)
    assert validator.final_scores[uid] == 1.0
