"""
factory.py — SLM Training Factory CLI entrypoint.

A modular, general-purpose fine-tuning factory for Small Language Models.
Supports arbitrary HuggingFace models, pluggable datasets, multiple training
methods, configurable optimization metrics, and metric-gated model versioning.

The original base model (e.g. Qwen/Qwen2.5-7B from HuggingFace) is NEVER
modified — only LoRA adapter weights are stored per version, and only when
the evaluation metric strictly improves.

Usage:
    # Train with a config file (recommended)
    python factory.py --config config/portuguese.yaml
    python factory.py --config config/portuguese_phase2.yaml

    # Override config fields from CLI
    python factory.py --config config/portuguese.yaml --model Qwen/Qwen2.5-7B --budget 1800

    # Quick check without training
    python factory.py --config config/portuguese.yaml --dry-run

    # List saved model versions and their metrics
    python factory.py versions --config config/portuguese.yaml

    # Merge LoRA adapter into base model (after training converges)
    python factory.py merge --config config/portuguese.yaml
    python factory.py merge --base Qwen/Qwen2.5-7B \\
        --adapter ./models/qwen-portuguese/best \\
        --output ./models/qwen-portuguese-v1-merged

    # List available datasets
    python factory.py list-datasets
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    base_path = Path(__file__).parent / "config" / "base.yaml"
    with open(base_path) as f:
        config = yaml.safe_load(f)

    with open(path) as f:
        override = yaml.safe_load(f)

    config = _deep_merge(config, override)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    import copy
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_cli_overrides(config: dict, args) -> dict:
    """Apply CLI argument overrides to config dict."""
    if getattr(args, "model", None):
        config["model"]["name"] = args.model
    if getattr(args, "dataset", None):
        config["dataset"]["name"] = args.dataset
    if getattr(args, "method", None):
        config["_method"] = args.method
    if getattr(args, "metric", None):
        config["metric"]["name"] = args.metric
    if getattr(args, "budget", None):
        config["training"]["budget_seconds"] = args.budget
    if getattr(args, "output_dir", None):
        config["training"]["output_dir"] = args.output_dir
    return config


# ---------------------------------------------------------------------------
# Trainer registry
# ---------------------------------------------------------------------------

def _get_trainer_class(method: str):
    method = method.lower()
    if method in ("qlora", "lora"):
        from trainers.qlora import QLoRATrainer
        return QLoRATrainer
    elif method == "full":
        raise NotImplementedError(
            "Full fine-tuning requires more VRAM than 6GB. "
            "Use method=qlora for 6GB VRAM targets."
        )
    else:
        raise ValueError(f"Unknown training method '{method}'. Available: qlora")


# ---------------------------------------------------------------------------
# Metric runner
# ---------------------------------------------------------------------------

def _run_metric(metric_name: str, model, tokenizer, eval_dataset, config: dict) -> float:
    max_seq_len = config["training"]["max_seq_len"]
    batch_size = config["training"]["batch_size"]
    text_col = config["dataset"].get("text_column", "text")

    if metric_name == "perplexity" or metric_name == "perplexity-pt":
        from metrics.perplexity import evaluate_perplexity_on_dataset
        return evaluate_perplexity_on_dataset(
            model, tokenizer, eval_dataset,
            text_column=text_col,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
        )
    elif metric_name == "bpb":
        from metrics.bpb import evaluate_bpb_on_dataset
        return evaluate_bpb_on_dataset(
            model, tokenizer, eval_dataset,
            text_column=text_col,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available: perplexity, bpb"
        )


# ---------------------------------------------------------------------------
# Main train command
# ---------------------------------------------------------------------------

def cmd_train(config: dict, dry_run: bool = False):
    """Load model, dataset, apply LoRA, train, evaluate, print results."""
    import torch

    t_start = time.time()

    model_cfg_dict = config["model"]
    lora_cfg_dict = config["lora"]
    train_cfg_dict = config["training"]
    dataset_cfg = config["dataset"]
    metric_name = config["metric"]["name"]
    method = config.get("_method", "qlora")

    # --- Model ---
    from models.registry import ModelConfig, load_model, estimate_vram_mb
    from models.adapters import LoraConfig, apply_lora

    model_cfg = ModelConfig.from_dict(model_cfg_dict)
    lora_cfg = LoraConfig.from_dict(lora_cfg_dict)

    if dry_run:
        vram_est = estimate_vram_mb(model_cfg.name, model_cfg.load_in_4bit)
        print(f"\n--- DRY RUN ---")
        print(f"Model:       {model_cfg.name}")
        print(f"Quantized:   {model_cfg.load_in_4bit} (4-bit NF4)")
        print(f"VRAM est.:   {vram_est:.0f} MB")
        print(f"LoRA rank:   r={lora_cfg.r}, alpha={lora_cfg.alpha}")
        print(f"Dataset:     {dataset_cfg['name']}")
        print(f"Method:      {method}")
        print(f"Metric:      {metric_name}")
        print(f"Budget:      {train_cfg_dict['budget_seconds']}s")
        print(f"Output dir:  {train_cfg_dict['output_dir']}")

        if vram_est > 6000:
            print(f"\nWARNING: Estimated VRAM {vram_est:.0f}MB exceeds 6GB target.")
            print("Consider: reduce max_seq_len, use smaller model, or enable load_in_4bit.")
        else:
            print(f"\nVRAM check OK ({vram_est:.0f}MB < 6000MB)")
        return

    model, tokenizer = load_model(model_cfg)
    model = apply_lora(model, lora_cfg)

    # --- Dataset ---
    from data.registry import load_dataset_by_name

    print(f"\nLoading dataset: {dataset_cfg['name']}")
    full_dataset = load_dataset_by_name(
        dataset_cfg["name"],
        max_samples=dataset_cfg.get("max_samples"),
    )

    # Train / val split
    val_split = dataset_cfg.get("val_split", 0.005)
    split = full_dataset.train_test_split(test_size=val_split, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"  Train: {len(train_dataset):,} | Val: {len(eval_dataset):,}")

    # --- Trainer ---
    from trainers.base import TrainingConfig
    TrainerClass = _get_trainer_class(method)

    train_cfg = TrainingConfig.from_dict(train_cfg_dict)
    trainer = TrainerClass(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=train_cfg,
    )

    # --- Model version store (metric-gated: only saves when metric improves) ---
    from versioning.model_store import ModelStore
    store_dir = config.get("versioning", {}).get(
        "store_dir",
        str(Path(train_cfg.output_dir).parent / "versions"),
    )
    lower_is_better = config["metric"].get("lower_is_better", True)
    store = ModelStore(store_dir, lower_is_better=lower_is_better)
    store.init(base_model_name=model_cfg.name, metric_name=metric_name)

    # --- Train ---
    result = trainer.train()

    # --- Final metric evaluation ---
    print(f"\nRunning final {metric_name} evaluation...")
    final_metric = _run_metric(metric_name, model, tokenizer, eval_dataset, config)

    t_end = time.time()

    # --- Try to promote: save version only if metric improved ---
    import json as _json
    promoted = store.try_promote(
        model=model,
        tokenizer=tokenizer,
        metric_value=final_metric,
        config=config,
        description=(
            f"r={lora_cfg.r} lr={train_cfg_dict.get('learning_rate')} "
            f"steps={result.num_steps}"
        ),
    )

    # --- Print structured output (parsed by autoresearch/loop.py) ---
    print(f"\n{'='*50}")
    print(f"metric_value:     {final_metric:.6f}")
    print(f"training_seconds: {result.training_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {result.peak_vram_mb:.1f}")
    print(f"num_steps:        {result.num_steps}")
    print(f"checkpoint:       {train_cfg.output_dir}")
    print(f"version_promoted: {promoted}")
    print(f"best_version:     {store.best_version or 'none'}")
    print(f"{'='*50}")

    store.print_summary()


# ---------------------------------------------------------------------------
# Merge command
# ---------------------------------------------------------------------------

def cmd_merge(args):
    """Merge a LoRA adapter into the base model and save a standalone model."""
    from models.registry import merge_adapter_into_base

    base = args.base
    adapter = args.adapter
    output = args.output

    if not base or not adapter or not output:
        # Try to derive from config
        if args.config:
            config = load_config(args.config)
            base = base or config["model"]["name"]
            adapter = adapter or config["training"]["output_dir"]
            output = output or str(
                Path(config["training"]["output_dir"]).parent.parent
                / ("models/" + Path(config["training"]["output_dir"]).name + "-merged")
            )

    if not base or not adapter or not output:
        print("Error: --base, --adapter, and --output are required (or provide --config)")
        sys.exit(1)

    merge_adapter_into_base(base, adapter, output)


# ---------------------------------------------------------------------------
# List commands
# ---------------------------------------------------------------------------

def cmd_list_datasets():
    from data.registry import list_datasets
    print("Available datasets:")
    for name in list_datasets():
        print(f"  {name}")


def cmd_versions(args):
    """Print the version history for a store."""
    from versioning.model_store import ModelStore
    from trainers.base import TrainingConfig

    config = load_config(args.config) if args.config else {}
    train_cfg_dict = config.get("training", {})
    output_dir = train_cfg_dict.get("output_dir", "./checkpoints")
    store_dir = config.get("versioning", {}).get(
        "store_dir",
        str(Path(output_dir).parent / "versions"),
    )
    metric_name = config.get("metric", {}).get("name", "metric")
    lower = config.get("metric", {}).get("lower_is_better", True)

    store = ModelStore(store_dir, lower_is_better=lower)
    if not (Path(store_dir) / ModelStore.REGISTRY_FILE).exists():
        print(f"No version store found at: {store_dir}")
        print("Run a training experiment first to create versions.")
        return
    store._load_registry()
    store.print_summary()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLM Training Factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- train (default) ---
    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument("--config", help="Path to stage config YAML")
    train_parser.add_argument("--model", help="HuggingFace model ID or local path (overrides config)")
    train_parser.add_argument("--dataset", help="Dataset name (overrides config)")
    train_parser.add_argument("--method", default="qlora", help="Training method: qlora (default)")
    train_parser.add_argument("--metric", help="Evaluation metric (overrides config)")
    train_parser.add_argument("--budget", type=int, help="Training budget in seconds (overrides config)")
    train_parser.add_argument("--output-dir", help="Checkpoint output directory")
    train_parser.add_argument("--dry-run", action="store_true", help="Check config without training")

    # Add train as default (no subcommand needed)
    parser.add_argument("--config", help="Path to stage config YAML")
    parser.add_argument("--model", help="HuggingFace model ID or local path")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--method", default="qlora", help="Training method: qlora")
    parser.add_argument("--metric", help="Evaluation metric")
    parser.add_argument("--budget", type=int, help="Training budget in seconds")
    parser.add_argument("--output-dir", help="Checkpoint output directory")
    parser.add_argument("--dry-run", action="store_true")

    # --- merge ---
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter into base model")
    merge_parser.add_argument("--config", help="Stage config YAML (for base/adapter/output defaults)")
    merge_parser.add_argument("--base", help="Base model ID or path")
    merge_parser.add_argument("--adapter", help="LoRA adapter directory")
    merge_parser.add_argument("--output", help="Output path for merged model")

    # --- list-datasets ---
    subparsers.add_parser("list-datasets", help="List registered datasets")

    # --- versions ---
    ver_parser = subparsers.add_parser("versions", help="Show model version history")
    ver_parser.add_argument("--config", help="Stage config YAML (to find store path)")
    ver_parser.add_argument("--store", help="Explicit path to version store directory")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "merge":
        cmd_merge(args)
        return

    if args.command == "list-datasets":
        cmd_list_datasets()
        return

    if args.command == "versions":
        cmd_versions(args)
        return

    # Default: train
    if not args.config:
        # Try to infer a reasonable default
        if Path("config/portuguese.yaml").exists():
            args.config = "config/portuguese.yaml"
            print(f"No --config specified, defaulting to {args.config}")
        else:
            print("Error: --config is required. Example: python factory.py --config config/portuguese.yaml")
            parser.print_help()
            sys.exit(1)

    config = load_config(args.config)
    config = _apply_cli_overrides(config, args)

    cmd_train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
