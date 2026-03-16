"""
tests/test_factory_config.py — Unit tests for factory.py config loading and CLI helpers.

Tests:
- load_config: base + stage merging
- _deep_merge: recursive dict merging
- _apply_cli_overrides: CLI arg → config dict
- VRAM dry-run logic
"""

import pytest
from pathlib import Path


ROOT = Path(__file__).parent.parent


class TestDeepMerge:

    def test_simple_override(self):
        from factory import _deep_merge
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        from factory import _deep_merge
        base = {"training": {"lr": 0.001, "batch": 2}}
        override = {"training": {"lr": 0.0001}}
        result = _deep_merge(base, override)
        assert result["training"]["lr"] == 0.0001
        assert result["training"]["batch"] == 2  # preserved

    def test_deep_merge_does_not_mutate_base(self):
        from factory import _deep_merge
        base = {"a": {"x": 1}}
        override = {"a": {"x": 99}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1  # original unchanged

    def test_list_is_replaced_not_merged(self):
        from factory import _deep_merge
        base = {"modules": ["a", "b"]}
        override = {"modules": ["c"]}
        result = _deep_merge(base, override)
        assert result["modules"] == ["c"]


class TestLoadConfig:

    def test_load_portuguese_config(self):
        from factory import load_config
        config = load_config(str(ROOT / "config" / "portuguese.yaml"))
        # Should have merged base + portuguese
        assert config["model"]["name"] == "Qwen/Qwen1.5-7B"
        assert config["dataset"]["name"] == "wikipedia-pt"
        assert config["metric"]["name"] == "perplexity"
        assert config["lora"]["r"] == 64

    def test_load_phase2_config(self):
        from factory import load_config
        config = load_config(str(ROOT / "config" / "portuguese_phase2.yaml"))
        assert config["dataset"]["name"] == "wikipedia-pt"
        assert config["lora"]["r"] == 64
        assert config["metric"]["name"] == "perplexity"

    def test_base_defaults_present(self):
        from factory import load_config
        config = load_config(str(ROOT / "config" / "portuguese.yaml"))
        # These come from base.yaml
        assert "gradient_checkpointing" in config["training"]
        assert config["training"]["gradient_checkpointing"] is True
        assert "bf16" in config["training"]


class TestApplyCliOverrides:

    def _make_args(self, **kwargs):
        """Create a simple namespace-like object."""
        import argparse
        ns = argparse.Namespace(**{
            "model": None, "dataset": None, "method": None,
            "metric": None, "budget": None, "output_dir": None,
        })
        for k, v in kwargs.items():
            setattr(ns, k, v)
        return ns

    def test_model_override(self):
        from factory import _apply_cli_overrides
        config = {"model": {"name": "Qwen/Qwen1.5-7B"}, "training": {}}
        args = self._make_args(model="Qwen/Qwen1.5-4B")
        result = _apply_cli_overrides(config, args)
        assert result["model"]["name"] == "Qwen/Qwen1.5-4B"

    def test_budget_override(self):
        from factory import _apply_cli_overrides
        config = {"model": {}, "training": {"budget_seconds": 3600}}
        args = self._make_args(budget=600)
        result = _apply_cli_overrides(config, args)
        assert result["training"]["budget_seconds"] == 600

    def test_none_values_not_applied(self):
        from factory import _apply_cli_overrides
        config = {"model": {"name": "original"}, "training": {}}
        args = self._make_args()  # all None
        result = _apply_cli_overrides(config, args)
        assert result["model"]["name"] == "original"
