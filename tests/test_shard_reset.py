from tplr.sharded_dataset import compute_shard_state


def test_compute_shard_state_no_reset():
    epoch, shard = compute_shard_state(25, 10, None)
    assert (epoch, shard) == (0, 2)


def test_compute_shard_state_with_reset_before_point():
    epoch, shard = compute_shard_state(20, 10, 30)
    assert (epoch, shard) == (0, 2)


def test_compute_shard_state_with_reset_after_point():
    epoch, shard = compute_shard_state(35, 10, 30)
    assert (epoch, shard) == (1, 0)

    epoch, shard = compute_shard_state(50, 10, 30)
    assert (epoch, shard) == (1, 2)
