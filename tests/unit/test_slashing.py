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
    validator.excluded_from_gather = set()
    validator.exclusion_start_window = {}

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
