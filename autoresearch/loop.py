"""
autoresearch/loop.py — Autonomous hyperparameter research loop.

Runs factory.py repeatedly, proposes config mutations, and keeps versions
only when the evaluation metric strictly improves.

Outputs:
  results/<run_tag>/
    progress.log    ← human-readable run report (append one block per experiment)
    results.tsv     ← machine-readable tab-separated log
    current_config.yaml  ← best config so far
    exp_0001/
      config.yaml   ← config used for this experiment
      run.log       ← raw factory.py output
    exp_0002/
      ...

Usage:
    # 1-hour total run, 12-min experiments
    python autoresearch/loop.py \\
        --config config/portuguese.yaml \\
        --total-budget 3600 \\
        --experiment-budget 720

    # Run until Ctrl+C, full experiment budget from config
    python autoresearch/loop.py --config config/portuguese.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


RESULTS_DIR = Path("results")
FACTORY_CMD = [sys.executable, "factory.py"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# TSV log (machine-readable)
# ---------------------------------------------------------------------------

def init_results_tsv(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = run_dir / "results.tsv"
    if not tsv_path.exists():
        with open(tsv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "timestamp", "experiment_id", "metric_value",
                "delta_vs_best", "training_seconds", "peak_vram_mb",
                "num_steps", "status", "description", "config_delta",
            ])
    return tsv_path


def log_tsv(tsv_path: Path, experiment_id: str, metric_value: float,
            delta_vs_best: float, training_seconds: float, peak_vram_mb: float,
            num_steps: int, status: str, description: str, config_delta: Dict):
    with open(tsv_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            datetime.now().isoformat(), experiment_id,
            f"{metric_value:.6f}", f"{delta_vs_best:+.6f}",
            f"{training_seconds:.1f}", f"{peak_vram_mb:.1f}",
            num_steps, status, description, json.dumps(config_delta),
        ])


# ---------------------------------------------------------------------------
# Human-readable progress log
# ---------------------------------------------------------------------------

class ProgressLog:
    """Writes a human-readable progress log alongside the TSV."""

    def __init__(self, path: Path, model: str, dataset: str, metric: str,
                 total_budget: Optional[int]):
        self.path = path
        self._write_header(model, dataset, metric, total_budget)

    def _write_header(self, model, dataset, metric, total_budget):
        budget_str = f"{total_budget}s ({total_budget//60} min)" if total_budget else "unlimited"
        with self.path.open("w", encoding="utf-8") as f:
            f.write("═" * 68 + "\n")
            f.write(f"  SLM AUTORESEARCH RUN\n")
            f.write(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Model:     {model}\n")
            f.write(f"  Dataset:   {dataset}\n")
            f.write(f"  Metric:    {metric}  (lower is better)\n")
            f.write(f"  Budget:    {budget_str}\n")
            f.write("═" * 68 + "\n\n")

    def experiment_start(self, exp_num: int, description: str,
                         config_delta: Dict, candidate_config: Dict,
                         elapsed_total: float, total_budget: Optional[int]):
        budget_left = ""
        if total_budget:
            remaining = max(0, total_budget - elapsed_total)
            budget_left = f"  │  {remaining/60:.0f} min left"

        with self.path.open("a", encoding="utf-8") as f:
            f.write("─" * 68 + "\n")
            f.write(
                f"  EXPERIMENT {exp_num}"
                f"  │  {datetime.now().strftime('%H:%M:%S')}"
                f"  │  {description}"
                f"{budget_left}\n"
            )
            f.write("─" * 68 + "\n")

            if config_delta:
                f.write(f"  Config delta:\n")
                for section, vals in config_delta.items():
                    if isinstance(vals, dict):
                        for k, v in vals.items():
                            f.write(f"    {section}.{k} = {v}\n")
                    else:
                        f.write(f"    {section} = {vals}\n")
            else:
                f.write(f"  Config:  (baseline — no changes)\n")

            tr = candidate_config.get("training", {})
            lo = candidate_config.get("lora", {})
            f.write(
                f"  Training:  budget={tr.get('budget_seconds')}s"
                f"  │  r={lo.get('r')}  α={lo.get('alpha')}"
                f"  │  lr={tr.get('learning_rate')}"
                f"  │  seq={tr.get('max_seq_len')}\n"
            )
            f.write(f"  Running...\n")

    def experiment_end(self, exp_num: int, metric_value: float,
                       best_before: Optional[float], status: str,
                       training_seconds: float, peak_vram_mb: float,
                       num_steps: int, elapsed_total: float):
        with self.path.open("a", encoding="utf-8") as f:
            if status == "fail":
                f.write(f"\n  ✗ FAILED  (check run.log for details)\n\n")
                return

            if status == "keep":
                marker = "✓ IMPROVED"
                if best_before is None:
                    delta_str = "(first result)"
                else:
                    delta = metric_value - best_before
                    pct = 100 * delta / best_before if best_before else 0
                    delta_str = f"{delta:+.4f}  ({pct:+.2f}%)  vs previous best"
            else:
                marker = "✗ no improvement"
                if best_before is not None:
                    delta = metric_value - best_before
                    pct = 100 * delta / best_before if best_before else 0
                    delta_str = f"{delta:+.4f}  ({pct:+.2f}%)  vs best {best_before:.4f}"
                else:
                    delta_str = ""

            vram_str = f"{peak_vram_mb:.0f} MB" if peak_vram_mb > 0 else "N/A (CPU)"

            f.write(f"\n  {marker}\n")
            f.write(f"  Perplexity:  {metric_value:.4f}  {delta_str}\n")
            f.write(f"  Steps:       {num_steps}\n")
            f.write(f"  Train time:  {training_seconds:.0f}s"
                    f"  │  VRAM peak: {vram_str}\n")
            f.write(f"  Wall time:   {elapsed_total:.0f}s elapsed"
                    f"  ({elapsed_total/60:.1f} min)\n")
            f.write("\n")

    def write_summary(self, history: List[Dict], best_metric: Optional[float],
                      total_elapsed: float):
        kept = [h for h in history if h.get("status") == "keep"]
        failed = [h for h in history if h.get("status") == "fail"]

        with self.path.open("a", encoding="utf-8") as f:
            f.write("═" * 68 + "\n")
            f.write("  FINAL SUMMARY\n")
            f.write("═" * 68 + "\n")
            f.write(f"  Experiments run:   {len(history)}\n")
            f.write(f"  Improvements:      {len(kept)}\n")
            f.write(f"  Failed:            {len(failed)}\n")
            f.write(f"  Total wall time:   {total_elapsed:.0f}s"
                    f"  ({total_elapsed/60:.1f} min)\n")
            if best_metric is not None:
                f.write(f"  Best perplexity:   {best_metric:.4f}\n")
            f.write("\n")

            if history:
                f.write(f"  {'Exp':<8} {'Metric':>12} {'Δ':>10} {'Steps':>7}"
                        f"  {'Status':<12}  Description\n")
                f.write(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*7}"
                        f"  {'─'*12}  {'─'*30}\n")
                first_metric = None
                for h in history:
                    if h.get("status") == "fail":
                        f.write(f"  {h['id']:<8} {'FAILED':>12}\n")
                        continue
                    mv = h.get("metric_value", float("nan"))
                    if first_metric is None:
                        delta_str = "baseline"
                        first_metric = mv
                    else:
                        d = mv - first_metric
                        delta_str = f"{d:+.4f}"
                    status_icon = "✓ kept" if h["status"] == "keep" else "✗ discard"
                    desc = h.get("description", "")[:30]
                    f.write(
                        f"  {h['id']:<8} {mv:>12.4f} {delta_str:>10}"
                        f" {h.get('num_steps', 0):>7}"
                        f"  {status_icon:<12}  {desc}\n"
                    )
            f.write("\n")
            f.write(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("═" * 68 + "\n")


# ---------------------------------------------------------------------------
# Experiment runner — streams output live to terminal + file
# ---------------------------------------------------------------------------

def run_experiment(config_path: str, run_log_path: str) -> Tuple[Dict, int]:
    """
    Run factory.py, streaming stdout to terminal in real time
    while also writing to run_log_path.
    """
    cmd = FACTORY_CMD + ["--config", config_path]
    print(f"\n  $ {' '.join(cmd)}\n  log → {run_log_path}\n")
    print("  " + "·" * 62)

    env = os.environ.copy()
    # Allow PyTorch to expand CUDA memory segments rather than pre-allocating
    # large chunks — reduces fragmentation on low-VRAM GPUs.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with open(run_log_path, "w", buffering=1) as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            # Strip HF progress bars for cleaner terminal output
            if "\r" not in line and line.strip():
                print("  " + line, end="", flush=True)
            log_f.write(line)
        proc.wait()

    print("  " + "·" * 62)
    result = _parse_factory_output(run_log_path)
    return result, proc.returncode


def _parse_factory_output(log_path: str) -> Dict:
    """Parse key=value lines printed by factory.py after training."""
    result = {}
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                for key in ["metric_value", "training_seconds", "total_seconds",
                            "peak_vram_mb", "num_steps"]:
                    if line.startswith(key + ":"):
                        try:
                            result[key] = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
    except FileNotFoundError:
        pass
    return result


# ---------------------------------------------------------------------------
# Hyperparameter mutation proposals
# ---------------------------------------------------------------------------

def _propose_config_delta(history: list, current_config: dict) -> Tuple[Dict, str]:
    """Propose a small config change based on experiment history."""
    import random

    training = current_config.get("training", {})

    candidates = [
        # LoRA rank
        ({"lora": {"r": 16, "alpha": 32}},   "LoRA rank=16 (more regularization)"),
        ({"lora": {"r": 32, "alpha": 64}},   "LoRA rank=32"),
        ({"lora": {"r": 64, "alpha": 128}},  "LoRA rank=64 (baseline)"),
        ({"lora": {"r": 128, "alpha": 256}}, "LoRA rank=128 (more capacity)"),
        # Learning rate — fine-grained low-LR options useful for continued training
        ({"training": {"learning_rate": 5e-5}}, "LR = 5e-5 (very conservative)"),
        ({"training": {"learning_rate": 7.5e-5}}, "LR = 7.5e-5"),
        ({"training": {"learning_rate": 1e-4}}, "LR = 1e-4 (conservative)"),
        ({"training": {"learning_rate": 2e-4}}, "LR = 2e-4 (baseline)"),
        ({"training": {"learning_rate": 3e-4}}, "LR = 3e-4 (aggressive)"),
        ({"training": {"learning_rate": 5e-4}}, "LR = 5e-4 (very aggressive)"),
        # Sequence length
        ({"training": {"max_seq_len": 512}},  "seq_len=512 (faster, more variety)"),
        ({"training": {"max_seq_len": 1024}}, "seq_len=1024 (baseline)"),
        # Gradient accumulation
        ({"training": {"gradient_accumulation_steps": 4}},  "grad_accum=4 (noisier gradients)"),
        ({"training": {"gradient_accumulation_steps": 16}}, "grad_accum=16 (smoother gradients)"),
        # Warmup
        ({"training": {"warmup_ratio": 0.01}}, "warmup=1%"),
        ({"training": {"warmup_ratio": 0.05}}, "warmup=5%"),
        # Dropout
        ({"lora": {"dropout": 0.0}}, "LoRA dropout=0 (no regularization)"),
        ({"lora": {"dropout": 0.1}}, "LoRA dropout=0.1"),
    ]

    tried = {json.dumps(h.get("config_delta", {}), sort_keys=True) for h in history}
    untried = [
        (d, desc) for d, desc in candidates
        if json.dumps(d, sort_keys=True) not in tried
    ]

    if not untried:
        lr = training.get("learning_rate", 2e-4)
        new_lr = lr * random.uniform(0.5, 2.0)
        delta = {"training": {"learning_rate": round(new_lr, 7)}}
        return delta, f"Random LR perturbation: {new_lr:.2e}"

    return untried[0]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    config_path: str,
    total_budget: Optional[int] = None,
    experiment_budget: Optional[int] = None,
    max_experiments: Optional[int] = None,
):
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(config_path)

    # Override per-experiment budget if specified
    if experiment_budget:
        base_config = deep_merge(base_config, {"training": {"budget_seconds": experiment_budget}})

    model_name = base_config.get("model", {}).get("name", "?")
    dataset_name = base_config.get("dataset", {}).get("name", "?")
    metric_name = base_config.get("metric", {}).get("name", "perplexity")

    tsv_path = init_results_tsv(run_dir)
    log_path = run_dir / "progress.log"
    progress = ProgressLog(log_path, model_name, dataset_name, metric_name, total_budget)

    wall_start = time.time()

    budget_str = f"{total_budget//60} min" if total_budget else "unlimited"
    exp_budget = base_config["training"]["budget_seconds"]

    print(f"\n{'═'*68}")
    print(f"  SLM AUTORESEARCH  —  {run_tag}")
    print(f"  Model:      {model_name}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Metric:     {metric_name}")
    print(f"  Total budget:      {budget_str}")
    print(f"  Per-experiment:    {exp_budget}s ({exp_budget//60} min)")
    print(f"  Results dir:       {run_dir}/")
    print(f"  Human log:         {log_path}")
    print(f"  TSV log:           {tsv_path}")
    print(f"{'═'*68}\n")

    current_config = copy.deepcopy(base_config)
    current_config_path = str(run_dir / "current_config.yaml")
    save_config(current_config, current_config_path)

    history: List[Dict] = []
    best_metric: Optional[float] = None
    experiment_id = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    try:
        while True:
            # Check total budget
            elapsed = time.time() - wall_start
            if total_budget and elapsed >= total_budget:
                print(f"\n  Total budget reached ({elapsed:.0f}s). Stopping.")
                break
            if max_experiments and experiment_id >= max_experiments:
                print(f"\n  Max experiments ({max_experiments}) reached. Stopping.")
                break
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n  {MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                      f"stopping to avoid a runaway loop. Check run.log for errors.")
                break

            experiment_id += 1
            exp_tag = f"exp_{experiment_id:04d}"
            exp_dir = run_dir / exp_tag
            exp_dir.mkdir(exist_ok=True)

            # Propose config for this experiment
            if experiment_id == 1:
                config_delta = {}
                description = "Baseline config"
                candidate_config = copy.deepcopy(current_config)
            else:
                config_delta, description = _propose_config_delta(history, current_config)
                candidate_config = deep_merge(current_config, config_delta)

            # Point output_dir into this experiment's subdir
            candidate_config = deep_merge(candidate_config, {
                "training": {"output_dir": str(exp_dir / "checkpoint")}
            })

            candidate_config_path = str(exp_dir / "config.yaml")
            save_config(candidate_config, candidate_config_path)

            elapsed = time.time() - wall_start
            remaining = max(0, total_budget - elapsed) if total_budget else None

            print(f"\n{'─'*68}")
            remaining_str = f"  │  {remaining/60:.0f} min left" if remaining else ""
            print(f"  EXPERIMENT {experiment_id}  │  {datetime.now().strftime('%H:%M:%S')}"
                  f"  │  {description}{remaining_str}")
            print(f"{'─'*68}")
            if config_delta:
                print(f"  Config delta: {json.dumps(config_delta)}")

            progress.experiment_start(
                experiment_id, description, config_delta,
                candidate_config, elapsed, total_budget
            )

            # Run
            run_log = str(exp_dir / "run.log")
            output, returncode = run_experiment(candidate_config_path, run_log)

            elapsed_after = time.time() - wall_start

            if returncode != 0 or "metric_value" not in output:
                consecutive_failures += 1
                print(f"\n  ✗ FAILED  (returncode={returncode}) — check {run_log}")
                print(f"  Consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")
                progress.experiment_end(
                    experiment_id, float("inf"), best_metric, "fail",
                    0, 0, 0, elapsed_after
                )
                log_tsv(tsv_path, exp_tag, float("inf"), 0, 0, 0, 0,
                        "fail", description, config_delta)
                history.append({"id": exp_tag, "config_delta": config_delta,
                                 "status": "fail", "description": description})
                continue

            consecutive_failures = 0  # reset on any success

            metric_value    = output["metric_value"]
            training_secs   = output.get("training_seconds", 0.0)
            peak_vram_mb    = output.get("peak_vram_mb", 0.0)
            num_steps       = int(output.get("num_steps", 0))

            best_before = best_metric

            if best_metric is None or metric_value < best_metric:
                status = "keep"
                best_metric = metric_value
                current_config = candidate_config
                save_config(current_config, current_config_path)
                _was = "n/a" if best_before is None else f"{best_before:.4f}"
                print(f"\n  ✓ IMPROVED  →  perplexity = {metric_value:.4f}"
                      f"  (was: {_was})")
            else:
                status = "discard"
                print(f"\n  ✗ No improvement  →  perplexity = {metric_value:.4f}"
                      f"  (best: {best_metric:.4f})")

            delta_vs_best = metric_value - best_metric if best_metric else 0.0

            progress.experiment_end(
                experiment_id, metric_value, best_before, status,
                training_secs, peak_vram_mb, num_steps, elapsed_after
            )
            log_tsv(tsv_path, exp_tag, metric_value, delta_vs_best,
                    training_secs, peak_vram_mb, num_steps,
                    status, description, config_delta)

            history.append({
                "id": exp_tag, "config_delta": config_delta,
                "status": status, "description": description,
                "metric_value": metric_value, "num_steps": num_steps,
            })

    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user after {experiment_id} experiments.")

    finally:
        total_elapsed = time.time() - wall_start
        progress.write_summary(history, best_metric, total_elapsed)

        print(f"\n{'═'*68}")
        print(f"  AUTORESEARCH COMPLETE")
        print(f"  Experiments:  {experiment_id}")
        print(f"  Best metric:  {best_metric:.4f}" if best_metric else "  No results.")
        print(f"  Wall time:    {total_elapsed:.0f}s  ({total_elapsed/60:.1f} min)")
        print(f"  Human log:    {log_path}")
        print(f"  TSV log:      {tsv_path}")
        print(f"{'═'*68}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SLM Factory autoresearch loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True,
                        help="Stage config YAML (e.g. config/portuguese.yaml)")
    parser.add_argument("--total-budget", type=int, default=None,
                        help="Stop the whole loop after this many seconds (e.g. 3600 for 1 hour)")
    parser.add_argument("--experiment-budget", type=int, default=None,
                        help="Override per-experiment budget_seconds from config")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Stop after N experiments regardless of time")
    args = parser.parse_args()

    run_loop(
        args.config,
        total_budget=args.total_budget,
        experiment_budget=args.experiment_budget,
        max_experiments=args.max_experiments,
    )


if __name__ == "__main__":
    main()
