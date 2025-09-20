import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import tplr  # your package provides tplr.logger, etc.

# ---- Small test doubles ------------------------------------------------------


@dataclass
class FakeBucket:
    name: str


class FakeS3:
    """
    Minimal async S3 client stub.

    listings maps a prefix -> list[(key, size)] that will be returned by list_objects_v2.
    """

    def __init__(self, listings: dict[str, list[tuple[str, int]]] | None = None):
        self.listings = listings or {}
        self.calls: dict[str, list[dict[str, Any]]] = {"list_objects_v2": []}

    async def list_objects_v2(self, **kwargs) -> dict[str, Any]:
        self.calls["list_objects_v2"].append(kwargs)
        prefix: str = kwargs["Prefix"]
        contents = [{"Key": k, "Size": sz} for (k, sz) in self.listings.get(prefix, [])]
        # no pagination in this stub
        return {"Contents": contents, "IsTruncated": False}


class FakeComms:
    """
    Minimal comms stub. It 'uploads' and 'downloads' by writing files under repo_root.
    """

    def __init__(
        self,
        *,
        own_bucket: FakeBucket,
        top_bucket: FakeBucket | None = None,
        repo_root: Path,
        s3_listings: dict[str, list[tuple[str, int]]] | None = None,
        latest_window: int | None = None,
    ):
        self.bucket = own_bucket
        self._own_bucket = own_bucket
        self._top_bucket = top_bucket
        self._latest_window = latest_window
        self._s3 = FakeS3(s3_listings)
        self.repo_root = repo_root

        self.uploads: list[tuple[str, Path]] = []
        self.downloads: list[str] = []

    async def _get_s3_client(self, _: FakeBucket) -> FakeS3:
        return self._s3

    async def _get_highest_stake_validator_bucket(self) -> tuple[FakeBucket, int]:
        if self._top_bucket is None:
            raise RuntimeError("no top bucket available")
        return self._top_bucket, 424242

    async def s3_put_object(self, *, key: str, file_path: str) -> dict[str, Any]:
        # Simulate upload by copying/moving file into repo_root/key (for visibility).
        src = Path(file_path)
        dst = self.repo_root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            # copy content; keep local files intact for other ranks in tests
            dst.write_bytes(src.read_bytes())
        else:
            dst.write_text("")  # empty placeholder
        self.uploads.append((key, src))
        return {"ok": True, "key": key}

    async def s3_get_object(
        self,
        *,
        key: str,
        bucket: FakeBucket,
        load_data: bool = False,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        # Pointer file: return a dict (what DCPCheckpointer expects).
        if key.endswith("/_LATEST.json") and self._latest_window is not None:
            return {"latest_window": int(self._latest_window)}
        # For data: create a local mirrored file so DCP can see it.
        dst = self.repo_root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_text("x")  # 1B placeholder
        self.downloads.append(key)
        return {"ok": True, "key": key}


# ---- Fixtures / helpers ------------------------------------------------------


@pytest.fixture(autouse=True)
def _quiet_logger(monkeypatch: pytest.MonkeyPatch):
    """Mute tplr.logger during tests."""

    class _L:
        def info(self, *_a, **_k):
            pass

        def debug(self, *_a, **_k):
            pass

        def warning(self, *_a, **_k):
            pass

        def warn(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    monkeypatch.setattr(tplr, "logger", _L())
    return _L()


@pytest.fixture
def dcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Load module and give a convenience wrapper to patch rank/world/barrier on-demand.
    """
    m = importlib.import_module("tplr.dcp_checkpoint")
    # neutralize barrier to avoid torch.distributed
    monkeypatch.setattr(m, "_barrier", lambda group=None: None)
    return m


def set_dist(
    monkeypatch: pytest.MonkeyPatch, mod: Any, *, rank: int, world: int
) -> None:
    """Patch helper functions in the module to emulate a rank/world_size."""
    monkeypatch.setattr(mod, "_rank", lambda group=None: int(rank))
    monkeypatch.setattr(mod, "_world", lambda group=None: int(world))


# ---- Tests -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_choose_read_bucket_prefers_highest(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    top = FakeBucket("top-bucket")
    comms = FakeComms(own_bucket=own, top_bucket=top, repo_root=tmp_path)

    set_dist(monkeypatch, dcp, rank=0, world=2)

    ckpt = DCPCheckpointer(comms, uid=1, version="v1", repo_root=tmp_path)
    b1 = await ckpt._choose_read_bucket(prefer_highest_staked=True)
    b2 = await ckpt._choose_read_bucket(prefer_highest_staked=False)

    assert b1.name == "top-bucket"
    assert b2.name == "own-bucket"


@pytest.mark.asyncio
async def test_discover_latest_uses_pointer_if_present(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    top = FakeBucket("top-bucket")

    comms = FakeComms(
        own_bucket=own,
        top_bucket=top,
        repo_root=tmp_path,
        latest_window=37,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version="vers", repo_root=tmp_path)

    latest = await ckpt._discover_latest(prefer_highest_staked=True)
    assert latest == 37


@pytest.mark.asyncio
async def test_discover_latest_falls_back_to_listing(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    top = FakeBucket("top-bucket")

    # Listing contains several windows; max should be picked.
    prefix = "checkpoints/verX/"
    listings = {
        prefix: [
            (f"{prefix}5/.metadata", 1),
            (f"{prefix}6/shard-a", 10),
            (f"{prefix}12/shard-b", 10),
            (f"{prefix}12/extra_metadata.json", 200),
        ]
    }
    comms = FakeComms(
        own_bucket=own,
        top_bucket=top,
        repo_root=tmp_path,
        s3_listings=listings,
        latest_window=None,  # pointer missing
    )

    set_dist(monkeypatch, dcp, rank=0, world=2)
    ckpt = DCPCheckpointer(comms, uid=9, version="verX", repo_root=tmp_path)

    latest = await ckpt._discover_latest(prefer_highest_staked=True)
    assert latest == 12


@pytest.mark.asyncio
async def test_upload_round_robin_assignment_per_rank(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Build a fake local checkpoint folder and assert each rank uploads its slice
    via round‑robin by filename; rank0 also uploads metadata and pointer.
    """
    from tplr.dcp_checkpoint import DCPCheckpointer, Layout

    world = 3
    window = 10
    set_dist(monkeypatch, dcp, rank=0, world=world)

    own = FakeBucket("own")

    # Pre-populate S3 listings to make rank 0 think all files are already uploaded
    prefix = f"checkpoints/v/{window}/"
    listings = {
        prefix: [
            (f"{prefix}__0_0.distcp", 1000),
            (f"{prefix}__0_1.distcp", 1000),
            (f"{prefix}__1_0.distcp", 1000),
            (f"{prefix}__1_1.distcp", 1000),
            (f"{prefix}__2_0.distcp", 1000),
            (f"{prefix}__2_1.distcp", 1000),
            (f"{prefix}.metadata", 100),
            (f"{prefix}extra_metadata.json", 200),
        ]
    }

    comms = FakeComms(own_bucket=own, repo_root=tmp_path, s3_listings=listings)
    ckpt = DCPCheckpointer(comms, uid=1, version="v", repo_root=tmp_path)
    layout = Layout("v", window)

    # Create local_dir with files
    local_dir = tmp_path / layout.prefix
    local_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    (local_dir / ".metadata").write_text("m")
    (local_dir / "extra_metadata.json").write_text(
        json.dumps({"world_size_at_save": world})
    )

    # Data shards with DCP-style names (so owner ranks can be identified)
    data_names = [
        "__0_0.distcp",
        "__0_1.distcp",  # rank 0 owns these
        "__1_0.distcp",
        "__1_1.distcp",  # rank 1 owns these
        "__2_0.distcp",
        "__2_1.distcp",  # rank 2 owns these
    ]
    for n in data_names:
        (local_dir / n).write_text("data-" + n)

    expected_by_rank: dict[int, set[str]] = {r: set() for r in range(world)}
    # Each rank owns files matching its rank number in the filename
    for name in data_names:
        # Extract owner rank from filename pattern __<rank>_<idx>.distcp
        import re

        m = re.match(r"__(\d+)_\d+\.distcp", name)
        if m:
            owner_rank = int(m.group(1))
            if owner_rank < world:
                expected_by_rank[owner_rank].add(f"{layout.prefix}/{name}")

    # Run upload separately for each emulated rank
    uploads_per_rank: dict[int, list[str]] = {}

    for r in range(world):
        set_dist(monkeypatch, dcp, rank=r, world=world)
        before = len(comms.uploads)
        await ckpt.upload(
            window=window,
            background=False,
            delete_local_on_success=False,
            initial_sleep_s=0.01,  # Skip the initial sleep
            pointer_poll_timeout_s=0.01,  # Very short timeout
            pointer_poll_interval_s=0.01,  # Fast polling
            mop_up_missing=False,  # Skip mop-up
        )
        new = [k for (k, _) in comms.uploads[before:]]
        uploads_per_rank[r] = new

    # Validate: each rank uploaded its expected data files
    for r in range(world):
        uploaded = {k for k in uploads_per_rank[r] if k.endswith(".distcp")}
        assert uploaded == expected_by_rank[r]

    # Rank 0 should also have uploaded metadata + pointer
    r0_uploads = set(uploads_per_rank[0])
    assert f"{layout.prefix}/.metadata" in r0_uploads
    assert f"{layout.prefix}/extra_metadata.json" in r0_uploads
    assert f"checkpoints/v/_LATEST.json" in r0_uploads


@pytest.mark.asyncio
async def test_download_distributed_round_robin_across_ranks(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Emulate distributed download: run ranks 1,2 first (grab their assignments),
    then rank 0 (grabs metadata + mop-up).
    """
    from tplr.dcp_checkpoint import DCPCheckpointer, Layout, _is_meta

    world = 3
    window = 7
    version = "vv"
    layout = Layout(version, window)
    prefix = f"{layout.prefix}/"

    # Compose a listing with meta + data files
    filenames = [
        ".metadata",
        "extra_metadata.json",
        "aa.pt",
        "bb.pt",
        "cc.pt",
        "dd.pt",
        "ee.pt",
        "ff.pt",
    ]
    listings = {prefix: [(f"{prefix}{n}", 10) for n in filenames]}

    own = FakeBucket("own")
    comms = FakeComms(own_bucket=own, repo_root=tmp_path, s3_listings=listings)
    ckpt = DCPCheckpointer(comms, uid=3, version=version, repo_root=tmp_path)

    # Compute expected per rank (round-robin over non-meta only)
    data_only = [n for n in filenames if not _is_meta(n)]
    sorted_data = sorted(data_only)
    expected_by_rank: dict[int, set[str]] = {
        r: {f"{prefix}{name}" for i, name in enumerate(sorted_data) if i % world == r}
        for r in range(world)
    }
    meta_keys = {f"{prefix}.metadata", f"{prefix}extra_metadata.json"}

    # Run r=1,2 first to avoid rank-0 mop-up stealing their work; then r=0.
    downloads: dict[int, set[str]] = {}
    for r in (1, 2, 0):
        set_dist(monkeypatch, dcp, rank=r, world=world)
        before = len(comms.downloads)
        local_dir = await ckpt.download_distributed(
            window=window, prefer_highest_staked=True
        )
        assert local_dir is not None
        new = set(comms.downloads[before:])
        downloads[r] = new

    # Ranks 1 and 2 download exactly their data assignments; rank 0 includes meta and any mop-up.
    assert downloads[1] == expected_by_rank[1]
    assert downloads[2] == expected_by_rank[2]
    assert meta_keys.issubset(downloads[0])
    # After r=1,2 ran, rank0 should not need to download their already-present files (may still download its own).
    assert expected_by_rank[0].issuperset(downloads[0] - meta_keys)


@pytest.mark.asyncio
async def test_download_and_load_uses_sidecar(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Ensure download_and_load reads extra_metadata.json and calls load_local with that window.
    """
    from tplr.dcp_checkpoint import DCPCheckpointer, Layout

    own = FakeBucket("own")
    comms = FakeComms(own_bucket=own, repo_root=tmp_path)
    ckpt = DCPCheckpointer(comms, uid=5, version="z", repo_root=tmp_path)

    # Build a fake local dir that download_* would return
    layout = Layout("z", 123)
    local_dir = tmp_path / layout.prefix
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "extra_metadata.json").write_text(
        json.dumps({"window": 123, "global_step": 456})
    )

    # Stub out the download method to return our local_dir
    async def fake_download(**_kwargs):
        return local_dir

    monkeypatch.setattr(ckpt, "download_distributed", fake_download)
    called: dict[str, Any] = {}

    def fake_load_local(
        *, model: Any, window: int, process_group: Any | None = None
    ) -> None:
        called["window"] = window

    monkeypatch.setattr(ckpt, "load_local", fake_load_local)

    got = await ckpt.download_and_load(
        model=object(), window=None, shared_fs=True, process_group=None
    )
    assert got == (123, 456)  # Now returns (window, global_step)
    assert called["window"] == 123


@pytest.mark.asyncio
async def test_cleanup_local_checkpoints(tmp_path: Path, monkeypatch):
    """Test that cleanup_local_checkpoints removes old checkpoints correctly."""

    # Mock _rank to return 0 (master)
    import src.tplr.dcp_checkpoint as dcp_module

    monkeypatch.setattr(dcp_module, "_rank", lambda: 0)

    # Create test checkpoint directories
    version = "test_version"
    checkpoint_base = tmp_path / "checkpoints" / version
    checkpoint_base.mkdir(parents=True)

    # Create multiple window directories
    windows = [100, 200, 300, 400, 500]
    for window in windows:
        window_dir = checkpoint_base / str(window)
        window_dir.mkdir()
        # Add a dummy file to make it look like a real checkpoint
        (window_dir / "checkpoint.pt").touch()

    # Also create a non-window directory that should be ignored
    (checkpoint_base / "not_a_window").mkdir()
    (checkpoint_base / "not_a_window" / "file.txt").touch()

    # Setup DCPCheckpointer
    comms = FakeComms(
        own_bucket=FakeBucket("test-bucket"),
        repo_root=tmp_path,
    )
    ckpt = tplr.DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    # Test keeping 2 latest checkpoints
    ckpt.cleanup_local_checkpoints(keep_latest=2)

    # Check that only the latest 2 windows remain (400, 500)
    remaining = sorted(
        [
            int(d.name)
            for d in checkpoint_base.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
    )
    assert remaining == [400, 500], f"Expected [400, 500] but got {remaining}"

    # Check that old windows were removed
    assert not (checkpoint_base / "100").exists()
    assert not (checkpoint_base / "200").exists()
    assert not (checkpoint_base / "300").exists()

    # Check that non-window directory was not touched
    assert (checkpoint_base / "not_a_window").exists()

    # Test keeping 1 latest checkpoint
    ckpt.cleanup_local_checkpoints(keep_latest=1)

    remaining = sorted(
        [
            int(d.name)
            for d in checkpoint_base.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
    )
    assert remaining == [500], f"Expected [500] but got {remaining}"

    # Test with no checkpoints to clean (only 1 exists, keep_latest=2)
    ckpt.cleanup_local_checkpoints(keep_latest=2)

    remaining = sorted(
        [
            int(d.name)
            for d in checkpoint_base.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
    )
    assert remaining == [500], f"Expected [500] but got {remaining}"

    # Test with non-existent checkpoint directory
    ckpt_nonexistent = tplr.DCPCheckpointer(
        comms, uid=1, version="nonexistent", repo_root=tmp_path
    )
    # Should not raise, just return early
    ckpt_nonexistent.cleanup_local_checkpoints(keep_latest=1)


@pytest.mark.asyncio
async def test_check_checkpoint_exists_complete(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists returns True for complete checkpoint."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    window = 42
    version = "test_v1"
    prefix = f"checkpoints/{version}/{window}/"

    # Complete checkpoint with all rank files for validator world size 4
    listings = {
        prefix: [
            (f"{prefix}.metadata", 100),
            (f"{prefix}extra_metadata.json", 200),
            (f"{prefix}__0_0.distcp", 1000),
            (f"{prefix}__1_0.distcp", 1000),
            (f"{prefix}__2_0.distcp", 1000),
            (f"{prefix}__3_0.distcp", 1000),
            (f"{prefix}some_other_file.pt", 500),
        ]
    }

    comms = FakeComms(
        own_bucket=own,
        repo_root=tmp_path,
        s3_listings=listings,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=False
    )
    assert result is True


@pytest.mark.asyncio
async def test_check_checkpoint_exists_missing_metadata(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists returns False when metadata is missing."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    window = 42
    version = "test_v1"
    prefix = f"checkpoints/{version}/{window}/"

    # Missing .metadata file
    listings = {
        prefix: [
            (f"{prefix}extra_metadata.json", 200),
            (f"{prefix}__0_0.distcp", 1000),
            (f"{prefix}__1_0.distcp", 1000),
            (f"{prefix}__2_0.distcp", 1000),
            (f"{prefix}__3_0.distcp", 1000),
        ]
    }

    comms = FakeComms(
        own_bucket=own,
        repo_root=tmp_path,
        s3_listings=listings,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=False
    )
    assert result is False


@pytest.mark.asyncio
async def test_check_checkpoint_exists_missing_sidecar(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists returns False when sidecar is missing."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    window = 42
    version = "test_v1"
    prefix = f"checkpoints/{version}/{window}/"

    # Missing extra_metadata.json
    listings = {
        prefix: [
            (f"{prefix}.metadata", 100),
            (f"{prefix}__0_0.distcp", 1000),
            (f"{prefix}__1_0.distcp", 1000),
            (f"{prefix}__2_0.distcp", 1000),
            (f"{prefix}__3_0.distcp", 1000),
        ]
    }

    comms = FakeComms(
        own_bucket=own,
        repo_root=tmp_path,
        s3_listings=listings,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=False
    )
    assert result is False


@pytest.mark.asyncio
async def test_check_checkpoint_exists_incomplete_ranks(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists returns False when rank files are incomplete."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    window = 42
    version = "test_v1"
    prefix = f"checkpoints/{version}/{window}/"

    # Missing rank 2 file
    listings = {
        prefix: [
            (f"{prefix}.metadata", 100),
            (f"{prefix}extra_metadata.json", 200),
            (f"{prefix}__0_0.distcp", 1000),
            (f"{prefix}__1_0.distcp", 1000),
            # rank 2 missing
            (f"{prefix}__3_0.distcp", 1000),
        ]
    }

    # Override FakeComms to return proper sidecar data
    class TestComms(FakeComms):
        async def s3_get_object(self, *, key: str, bucket: FakeBucket, **kwargs):
            if "extra_metadata.json" in key:
                return {"world_size_at_save": 4}  # Expects 4 ranks
            return await super().s3_get_object(key=key, bucket=bucket, **kwargs)

    comms = TestComms(
        own_bucket=own,
        repo_root=tmp_path,
        s3_listings=listings,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=False
    )
    assert result is False


@pytest.mark.asyncio
async def test_check_checkpoint_exists_empty_directory(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists returns False for empty directory."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    window = 42
    version = "test_v1"
    prefix = f"checkpoints/{version}/{window}/"

    # Empty directory
    listings = {prefix: []}

    comms = FakeComms(
        own_bucket=own,
        repo_root=tmp_path,
        s3_listings=listings,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=False
    )
    assert result is False


@pytest.mark.asyncio
async def test_check_checkpoint_exists_with_highest_staked(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists works with highest-staked validator's bucket."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    top = FakeBucket("top-bucket")
    window = 42
    version = "test_v1"
    prefix = f"checkpoints/{version}/{window}/"

    # Complete checkpoint
    listings = {
        prefix: [
            (f"{prefix}.metadata", 100),
            (f"{prefix}extra_metadata.json", 200),
            (f"{prefix}__0_0.distcp", 1000),
            (f"{prefix}__1_0.distcp", 1000),
            (f"{prefix}__2_0.distcp", 1000),
            (f"{prefix}__3_0.distcp", 1000),
        ]
    }

    comms = FakeComms(
        own_bucket=own,
        top_bucket=top,
        repo_root=tmp_path,
        s3_listings=listings,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    # Should use top bucket when prefer_highest_staked=True
    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=True
    )
    assert result is True


@pytest.mark.asyncio
async def test_check_checkpoint_exists_handles_exception(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists handles exceptions gracefully."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    window = 42
    version = "test_v1"

    class FailingS3:
        async def list_objects_v2(self, **kwargs):
            raise RuntimeError("S3 connection failed")

    class FailingComms(FakeComms):
        async def _get_s3_client(self, _):
            return FailingS3()

    comms = FailingComms(
        own_bucket=own,
        repo_root=tmp_path,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    # Should return False on exception
    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=False
    )
    assert result is False


@pytest.mark.asyncio
async def test_check_checkpoint_exists_multiple_metadata_files(
    dcp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test check_checkpoint_exists handles multiple metadata files correctly."""
    from tplr.dcp_checkpoint import DCPCheckpointer

    own = FakeBucket("own-bucket")
    window = 42
    version = "test_v1"
    prefix = f"checkpoints/{version}/{window}/"

    # Multiple metadata files (e.g., app.metadata, optimizer.metadata)
    listings = {
        prefix: [
            (f"{prefix}.metadata", 100),
            (f"{prefix}app.metadata", 100),
            (f"{prefix}optimizer.metadata", 100),
            (f"{prefix}extra_metadata.json", 200),
            (f"{prefix}__0_0.distcp", 1000),
            (f"{prefix}__1_0.distcp", 1000),
            (f"{prefix}__2_0.distcp", 1000),
            (f"{prefix}__3_0.distcp", 1000),
        ]
    }

    comms = FakeComms(
        own_bucket=own,
        repo_root=tmp_path,
        s3_listings=listings,
    )

    set_dist(monkeypatch, dcp, rank=0, world=1)
    ckpt = DCPCheckpointer(comms, uid=1, version=version, repo_root=tmp_path)

    result = await ckpt.check_checkpoint_exists(
        window=window, prefer_highest_staked=False
    )
    assert result is True
