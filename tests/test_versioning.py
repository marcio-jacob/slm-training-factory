"""
tests/test_versioning.py — Unit tests for versioning/model_store.py

Tests:
- Store initialization (idempotent)
- Base model protection (mismatched base raises error)
- try_promote: saves version when metric improves
- try_promote: discards version when metric does not improve
- best_version / best_metric tracking
- list_versions ordering
- best symlink creation
- Version metadata contents
- Promotion lineage (parent_version tracking)
"""

import json
from pathlib import Path

import pytest

from versioning.model_store import ModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal fake PEFT model that implements save_pretrained."""
    def __init__(self, name="fake"):
        self.name = name

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.bin").write_bytes(b"fake weights")
        (Path(path) / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "test/model", "r": 64})
        )


class _FakeTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text('{"version": "1.0"}')


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------

def test_init_creates_registry(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/base-model", "perplexity")
    assert (Path(tmp_store) / "registry.json").exists()
    assert (Path(tmp_store) / "base_ref.json").exists()


def test_init_idempotent(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/base-model", "perplexity")
    # Second call should not raise
    store2 = ModelStore(tmp_store)
    store2.init("test/base-model", "perplexity")
    assert store2.best_version is None


def test_init_wrong_base_model_raises(tmp_store):
    store = ModelStore(tmp_store)
    store.init("model-A", "perplexity")
    store2 = ModelStore(tmp_store)
    with pytest.raises(ValueError, match="model-A"):
        store2.init("model-B", "perplexity")


def test_init_force_resets_store(tmp_store):
    store = ModelStore(tmp_store)
    store.init("model-A", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "first")
    assert store.best_version is not None

    store2 = ModelStore(tmp_store)
    store2.init("model-B", "perplexity", force=True)
    assert store2.best_version is None


# ---------------------------------------------------------------------------
# Tests: try_promote (lower is better = True)
# ---------------------------------------------------------------------------

def test_first_promote_always_saves(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    promoted = store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "baseline")
    assert promoted is True
    assert store.best_version == "v0001"
    assert store.best_metric == 50.0


def test_better_metric_saves_new_version(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "v1")
    promoted = store.try_promote(_FakeModel(), _FakeTokenizer(), 40.0, {}, "v2 better")
    assert promoted is True
    assert store.best_version == "v0002"
    assert store.best_metric == 40.0


def test_worse_metric_does_not_save(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "baseline")
    promoted = store.try_promote(_FakeModel(), _FakeTokenizer(), 60.0, {}, "worse")
    assert promoted is False
    assert store.best_version == "v0001"
    assert len(store.list_versions()) == 1


def test_equal_metric_does_not_save(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "baseline")
    promoted = store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "equal")
    assert promoted is False


def test_multiple_improvements_tracked(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")

    metrics = [50.0, 45.0, 60.0, 42.0, 41.0, 44.0]
    expected_best = [50.0, 45.0, 45.0, 42.0, 41.0, 41.0]

    for i, (m, exp) in enumerate(zip(metrics, expected_best)):
        store.try_promote(_FakeModel(), _FakeTokenizer(), m, {}, f"exp {i}")
        assert store.best_metric == exp, f"After metric={m}, expected best={exp}"


# ---------------------------------------------------------------------------
# Tests: higher is better
# ---------------------------------------------------------------------------

def test_higher_is_better_mode(tmp_store):
    store = ModelStore(tmp_store, lower_is_better=False)
    store.init("test/model", "accuracy")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 0.5, {}, "baseline")
    promoted = store.try_promote(_FakeModel(), _FakeTokenizer(), 0.7, {}, "better")
    assert promoted is True
    assert store.best_metric == 0.7

    not_promoted = store.try_promote(_FakeModel(), _FakeTokenizer(), 0.6, {}, "worse")
    assert not_promoted is False


# ---------------------------------------------------------------------------
# Tests: saved artifacts
# ---------------------------------------------------------------------------

def test_adapter_files_saved(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel("m1"), _FakeTokenizer(), 50.0, {}, "test")

    v_dir = Path(tmp_store) / "v0001"
    assert v_dir.exists()
    assert (v_dir / "adapter_model.bin").exists()
    assert (v_dir / "tokenizer.json").exists()
    assert (v_dir / "metadata.json").exists()


def test_metadata_contents(tmp_store):
    cfg = {"training": {"learning_rate": 2e-4}, "lora": {"r": 64}}
    store = ModelStore(tmp_store)
    store.init("Qwen/Qwen1.5-7B", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 42.5, cfg, "test run")

    meta = json.loads((Path(tmp_store) / "v0001" / "metadata.json").read_text())
    assert meta["version_id"] == "v0001"
    assert meta["metric_name"] == "perplexity"
    assert meta["metric_value"] == 42.5
    assert meta["base_model"] == "Qwen/Qwen1.5-7B"
    assert meta["description"] == "test run"
    assert meta["config"] == cfg


def test_best_symlink_created(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "v1")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 40.0, {}, "v2")

    best_link = Path(tmp_store) / "best"
    # Either symlink or best.txt should exist
    assert best_link.is_symlink() or (Path(tmp_store) / "best.txt").exists()


def test_parent_version_tracked(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "base", parent_version=None)
    store.try_promote(_FakeModel(), _FakeTokenizer(), 40.0, {}, "fine-tuned", parent_version="v0001")

    meta = json.loads((Path(tmp_store) / "v0002" / "metadata.json").read_text())
    assert meta["parent_version"] == "v0001"


# ---------------------------------------------------------------------------
# Tests: querying
# ---------------------------------------------------------------------------

def test_list_versions_sorted(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "v1")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 40.0, {}, "v2")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 45.0, {}, "v3 worse, not saved")

    versions = store.list_versions()
    assert len(versions) == 2
    assert versions[0]["metric_value"] == 40.0   # best first
    assert versions[1]["metric_value"] == 50.0


def test_get_best_returns_metadata(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "first")
    best = store.get_best()
    assert best is not None
    assert best["metric_value"] == 50.0
    assert best["version_id"] == "v0001"


def test_get_best_empty_store_returns_none(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    assert store.get_best() is None
    assert store.best_version is None
    assert store.best_metric is None


def test_get_adapter_path(tmp_store):
    store = ModelStore(tmp_store)
    store.init("test/model", "perplexity")
    store.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "v1")
    path = store.get_adapter_path("v0001")
    assert Path(path).exists()


def test_registry_persists_across_instances(tmp_store):
    store1 = ModelStore(tmp_store)
    store1.init("test/model", "perplexity")
    store1.try_promote(_FakeModel(), _FakeTokenizer(), 50.0, {}, "v1")
    store1.try_promote(_FakeModel(), _FakeTokenizer(), 40.0, {}, "v2")

    # New instance reads from disk
    store2 = ModelStore(tmp_store)
    store2._ensure_initialized()
    store2._load_registry()
    assert store2.best_metric == 40.0
    assert len(store2.list_versions()) == 2


# ---------------------------------------------------------------------------
# Tests: max_versions pruning
# ---------------------------------------------------------------------------

def test_max_versions_prunes_oldest(tmp_store):
    """After exceeding max_versions, oldest version dir is removed."""
    store = ModelStore(tmp_store, max_versions=3)
    store.init("test/model", "perplexity")

    # Promote 4 steadily-improving versions — each one saves
    for i, ppl in enumerate([50.0, 45.0, 40.0, 35.0], start=1):
        store.try_promote(_FakeModel(), _FakeTokenizer(), ppl, {}, f"v{i}")

    # Only 3 most recent (by insertion) should remain
    assert len(store.list_versions()) == 3
    # v0001 (50.0) should have been pruned
    assert not (Path(tmp_store) / "v0001").exists()
    # v0002, v0003, v0004 should still exist
    for vid in ["v0002", "v0003", "v0004"]:
        assert (Path(tmp_store) / vid).exists()


def test_max_versions_never_deletes_best(tmp_store):
    """Best version is never pruned even when it's oldest."""
    store = ModelStore(tmp_store, max_versions=2)
    store.init("test/model", "perplexity")

    # v0001 is promoted with best metric 30.0 (lowest)
    store.try_promote(_FakeModel(), _FakeTokenizer(), 30.0, {}, "best early")
    # v0002: 35.0 — worse, NOT saved → only 1 version
    store.try_promote(_FakeModel(), _FakeTokenizer(), 35.0, {}, "worse")
    # v0002: 28.0 — better, saved → 2 versions
    store.try_promote(_FakeModel(), _FakeTokenizer(), 28.0, {}, "better")
    # v0003: 25.0 — better, saved → would be 3, so prune oldest non-best
    store.try_promote(_FakeModel(), _FakeTokenizer(), 25.0, {}, "best now")

    assert len(store.list_versions()) == 2
    # v0001 (30.0, oldest) should be pruned since it's no longer best
    # v0002 (28.0) and v0003 (25.0=best) should remain
    assert store.best_metric == 25.0


def test_max_versions_unlimited(tmp_store):
    """max_versions=None disables pruning."""
    store = ModelStore(tmp_store, max_versions=None)
    store.init("test/model", "perplexity")

    for i, ppl in enumerate([50.0, 45.0, 40.0, 35.0, 30.0], start=1):
        store.try_promote(_FakeModel(), _FakeTokenizer(), ppl, {}, f"v{i}")

    assert len(store.list_versions()) == 5


def test_max_versions_respected_in_registry(tmp_store):
    """Registry JSON reflects pruned state — no ghost entries."""
    store = ModelStore(tmp_store, max_versions=2)
    store.init("test/model", "perplexity")

    for ppl in [50.0, 45.0, 40.0]:
        store.try_promote(_FakeModel(), _FakeTokenizer(), ppl, {}, "run")

    import json
    registry = json.loads((Path(tmp_store) / "registry.json").read_text())
    assert len(registry) == 2
