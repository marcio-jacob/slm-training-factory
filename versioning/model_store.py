"""
versioning/model_store.py — Metric-gated model version store.

Manages a versioned history of fine-tuned model checkpoints.
A new version is only committed when the evaluation metric strictly improves.
The original base model (Qwen 7B from HuggingFace) is NEVER modified —
only LoRA adapter weights are stored per version.

Directory layout:
    models/
    ├── registry.json        ← version index (human-readable)
    ├── base_ref.json        ← pointer to the original base model (HF ID or local path)
    ├── v0001/
    │   ├── adapter_model.safetensors   (or adapter_model.bin)
    │   ├── adapter_config.json
    │   ├── tokenizer files
    │   └── metadata.json    ← {version, metric_name, metric_value, timestamp, config, parent}
    ├── v0002/
    │   └── ...
    └── best -> v0002/       ← symlink to the current best version

Usage:
    store = ModelStore("./models")
    store.init(base_model_name="Qwen/Qwen1.5-7B", metric_name="perplexity")

    # After training:
    promoted = store.try_promote(
        model=peft_model,
        tokenizer=tokenizer,
        metric_value=42.3,
        config=config_dict,
        description="LoRA r=64, LR=2e-4, Wikipedia PT"
    )
    if promoted:
        print(f"New best: v{store.best_version}")

    # To load the best checkpoint:
    version = store.get_best()
    adapter_path = store.get_adapter_path(version["version_id"])
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VersionMetadata:
    version_id: str            # e.g. "v0001"
    metric_name: str           # e.g. "perplexity"
    metric_value: float        # The value at the time of promotion
    timestamp: str             # ISO 8601
    base_model: str            # Original HF model ID or path (never modified)
    parent_version: Optional[str]   # ID of the version this was fine-tuned from
    config: Dict[str, Any]     # The training config used
    description: str           # Human-readable summary
    adapter_path: str          # Relative path to adapter within store root

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VersionMetadata":
        return cls(**d)


class ModelStore:
    """
    Manages versioned LoRA adapter checkpoints.

    Key guarantee: a new version is only written when metric_value
    is strictly better than all previous versions (lower is better
    for perplexity/BPB; configurable via lower_is_better).
    """

    REGISTRY_FILE = "registry.json"
    BASE_REF_FILE = "base_ref.json"
    BEST_LINK = "best"

    def __init__(self, store_root: str, lower_is_better: bool = True, max_versions: int = 3):
        self.root = Path(store_root)
        self.lower_is_better = lower_is_better
        self.max_versions = max_versions
        self._registry: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, base_model_name: str, metric_name: str, force: bool = False):
        """
        Initialize the store for a given base model and metric.
        Safe to call multiple times (idempotent unless force=True).

        Args:
            base_model_name: HuggingFace model ID or local path of the BASE model.
                             This model is never modified — only adapters are stored.
            metric_name: The optimization metric (e.g. "perplexity").
            force: If True, wipe existing store and start fresh.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        if force and self.registry_path.exists():
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)

        base_ref = {"base_model": base_model_name, "metric_name": metric_name}
        if not self.base_ref_path.exists():
            self.base_ref_path.write_text(json.dumps(base_ref, indent=2))
            print(f"Model store initialized at: {self.root}")
            print(f"  Base model (protected): {base_model_name}")
            print(f"  Metric:                  {metric_name}")
        else:
            # Validate consistency
            existing = json.loads(self.base_ref_path.read_text())
            if existing["base_model"] != base_model_name:
                raise ValueError(
                    f"Store was initialized with base model '{existing['base_model']}' "
                    f"but now given '{base_model_name}'. "
                    f"Use force=True to reset, or use the correct base model."
                )

        if not self.registry_path.exists():
            self._registry = []
            self._save_registry()
        else:
            self._load_registry()

    # ------------------------------------------------------------------
    # Core: try_promote
    # ------------------------------------------------------------------

    def try_promote(
        self,
        model,
        tokenizer,
        metric_value: float,
        config: Dict[str, Any],
        description: str = "",
        parent_version: Optional[str] = None,
    ) -> bool:
        """
        Attempt to save a new version. Only commits if metric_value is
        strictly better than the current best.

        Args:
            model: PEFT model (LoRA adapter) to save
            tokenizer: Tokenizer to save alongside the adapter
            metric_value: The evaluation metric for this checkpoint
            config: The training config dict used to produce this model
            description: Human-readable label
            parent_version: Version ID this was fine-tuned from (for lineage tracking)

        Returns:
            True if promoted (metric improved), False if discarded.
        """
        self._ensure_initialized()

        best = self.best_metric
        if best is not None:
            if self.lower_is_better and metric_value >= best:
                print(f"  Version NOT saved: {metric_value:.6f} >= best {best:.6f} (no improvement)")
                return False
            elif not self.lower_is_better and metric_value <= best:
                print(f"  Version NOT saved: {metric_value:.6f} <= best {best:.6f} (no improvement)")
                return False

        # Generate version ID
        version_id = f"v{len(self._registry) + 1:04d}"
        version_dir = self.root / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights + tokenizer
        print(f"  Saving new best version {version_id} (metric={metric_value:.6f})...")
        model.save_pretrained(str(version_dir))
        tokenizer.save_pretrained(str(version_dir))

        # Save metadata
        base_model = json.loads(self.base_ref_path.read_text())["base_model"]
        meta = VersionMetadata(
            version_id=version_id,
            metric_name=self._metric_name,
            metric_value=metric_value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            base_model=base_model,
            parent_version=parent_version,
            config=config,
            description=description,
            adapter_path=str(version_id),
        )
        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta.as_dict(), indent=2))

        # Update registry
        self._registry.append(meta.as_dict())
        self._save_registry()

        # Update best symlink
        self._update_best_link(version_id)

        print(f"  ✓ Promoted to {version_id} — new best {self._metric_name}={metric_value:.6f}")

        # Enforce max_versions: prune oldest non-best versions
        if self.max_versions is not None:
            self._prune_old_versions()

        return True

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    @property
    def best_version(self) -> Optional[str]:
        """Version ID of the current best checkpoint, or None if empty."""
        versions = self._kept_versions()
        if not versions:
            return None
        key = min if self.lower_is_better else max
        best = key(versions, key=lambda v: v["metric_value"])
        return best["version_id"]

    @property
    def best_metric(self) -> Optional[float]:
        """Metric value of the current best checkpoint, or None if empty."""
        versions = self._kept_versions()
        if not versions:
            return None
        key = min if self.lower_is_better else max
        return key(v["metric_value"] for v in versions)

    def get_best(self) -> Optional[Dict[str, Any]]:
        """Return the metadata dict of the current best version."""
        vid = self.best_version
        if vid is None:
            return None
        return self.get_version(vid)

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata dict for a specific version."""
        for v in self._registry:
            if v["version_id"] == version_id:
                return v
        return None

    def get_adapter_path(self, version_id: str) -> str:
        """Return the absolute filesystem path to a version's adapter directory."""
        return str(self.root / version_id)

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all saved versions sorted by metric value (best first)."""
        versions = list(self._registry)
        reverse = not self.lower_is_better
        return sorted(versions, key=lambda v: v["metric_value"], reverse=reverse)

    def print_summary(self):
        """Print a human-readable summary of all saved versions."""
        versions = self.list_versions()
        if not versions:
            print(f"Model store at {self.root}: no versions saved yet.")
            return

        print(f"\nModel Store: {self.root}")
        print(f"Base model (protected): {json.loads(self.base_ref_path.read_text())['base_model']}")
        print(f"Metric: {self._metric_name} ({'lower' if self.lower_is_better else 'higher'} is better)")
        print(f"{'─'*70}")
        print(f"{'Version':<10} {'Metric':>12} {'Timestamp':<25} Description")
        print(f"{'─'*70}")
        best_id = self.best_version
        for v in versions:
            marker = " ★" if v["version_id"] == best_id else ""
            print(
                f"{v['version_id']:<10} "
                f"{v['metric_value']:>12.6f} "
                f"{v['timestamp'][:19]:<25} "
                f"{v.get('description', '')[:30]}"
                f"{marker}"
            )
        print(f"{'─'*70}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def registry_path(self) -> Path:
        return self.root / self.REGISTRY_FILE

    @property
    def base_ref_path(self) -> Path:
        return self.root / self.BASE_REF_FILE

    @property
    def _metric_name(self) -> str:
        if not self.base_ref_path.exists():
            return "metric"
        return json.loads(self.base_ref_path.read_text()).get("metric_name", "metric")

    def _kept_versions(self) -> List[Dict[str, Any]]:
        return [v for v in self._registry if v.get("metric_value") is not None]

    def _save_registry(self):
        self.registry_path.write_text(json.dumps(self._registry, indent=2))

    def _load_registry(self):
        self._registry = json.loads(self.registry_path.read_text())

    def _update_best_link(self, version_id: str):
        """Update the 'best' symlink to point at the given version."""
        link_path = self.root / self.BEST_LINK
        target = Path(version_id)  # relative symlink

        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        try:
            link_path.symlink_to(target)
        except OSError:
            # Windows or permission issue — write a plain text pointer instead
            (self.root / "best.txt").write_text(version_id)

    def _prune_old_versions(self):
        """Delete oldest versions (by insertion order) when count exceeds max_versions.
        The current best version is never deleted regardless of age."""
        if len(self._registry) <= self.max_versions:
            return

        best_id = self.best_version
        # Walk registry in insertion order; collect candidates for deletion
        to_delete = []
        for v in self._registry:
            if len(self._registry) - len(to_delete) <= self.max_versions:
                break
            if v["version_id"] != best_id:
                to_delete.append(v["version_id"])

        for vid in to_delete:
            version_dir = self.root / vid
            if version_dir.exists():
                shutil.rmtree(version_dir)
                print(f"  Pruned old version {vid} (max_versions={self.max_versions})")
            self._registry = [v for v in self._registry if v["version_id"] != vid]

        self._save_registry()

    def _ensure_initialized(self):
        if not self.base_ref_path.exists():
            raise RuntimeError(
                "ModelStore not initialized. Call store.init(base_model_name, metric_name) first."
            )
        if not self._registry and self.registry_path.exists():
            self._load_registry()
