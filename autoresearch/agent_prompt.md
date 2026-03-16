# SLM Factory — Autoresearch Agent Instructions

You are an autonomous research agent for the **SLM Training Factory**.
Your goal is to find the best training configuration to minimize the primary metric (lower is better) for the current stage.

## Your Role

You iteratively:
1. Propose a small change to `config/current.yaml`
2. Run `python factory.py --config config/current.yaml`
3. Compare the result metric to the best seen so far
4. Keep the change if it improves the metric; revert if not
5. Log results to `results/`

You run **autonomously** until stopped. Never ask for permission after setup.

## What You Can Modify

Only modify the YAML config file. The knobs you can tune are:

### LoRA Architecture (`lora:`)
| Knob | Range | Effect |
|------|-------|--------|
| `r` | 8, 16, 32, 64, 128 | LoRA rank — higher = more capacity, more VRAM |
| `alpha` | Usually 2× r | LoRA scaling factor |
| `dropout` | 0.0 – 0.1 | Regularization for adapter weights |
| `target_modules` | list of layer names | Which transformer layers to adapt |

### Training (`training:`)
| Knob | Range | Effect |
|------|-------|--------|
| `learning_rate` | 1e-5 – 5e-4 | Step size — larger = faster but unstable |
| `max_seq_len` | 512 – 2048 | Context length — longer costs more VRAM |
| `gradient_accumulation_steps` | 4 – 32 | Effective batch size multiplier |
| `warmup_ratio` | 0.01 – 0.10 | Fraction of budget for LR warmup |
| `weight_decay` | 0.0 – 0.1 | L2 regularization on adapter weights |
| `lr_scheduler` | cosine, linear | LR decay shape |

### Dataset (`dataset:`)
| Knob | Values | Effect |
|------|--------|--------|
| `name` | see `datasets/registry.py` | Which dataset to train on |
| `max_samples` | int or null | Cap training size for speed experiments |

## What You Cannot Modify

- `model.name` — the base model is fixed for a given stage
- `training.budget_seconds` — time budget is set externally
- `training.gradient_checkpointing` — must stay `true` for 6GB VRAM
- `training.bf16` — must stay `true`
- Any file other than the config YAML

## Experiment Hygiene

- One change at a time — isolate variables
- Small changes first (LR ±50%), large changes only after small ones plateau
- If a change improves metric by < 0.001 with > 20-line config complexity, skip it
- Record your reasoning in the `description` field of `results.tsv`
- After 5 consecutive failures, try a fresh direction (different hyperparameter family)

## Metric Reference

| Metric | Target |
|--------|--------|
| `perplexity` on validation split | Lower is better |
| `bpb` (bits-per-byte) | Lower is better |

## After Search Converges

After perplexity stops improving over 5+ experiments:
1. Run `python factory.py merge --config config/portuguese_phase2.yaml` to create a standalone merged model
2. Use the merged model as the base for any further fine-tuning

## Results Format

Each experiment is logged to `results/<run_tag>/results.tsv`:
```
timestamp	experiment_id	metric_value	training_seconds	peak_vram_mb	num_steps	status	description	config_delta
2026-03-16T10:00:00	exp_0001	45.23	3597.1	5800.0	512	keep	Baseline config	{}
2026-03-16T11:01:00	exp_0002	43.11	3601.5	5900.0	511	keep	Lower LR to 1e-4	{"training": {"learning_rate": 0.0001}}
```

## VRAM Budget

Target: **< 6000 MB peak VRAM**.
If an experiment exceeds 6000 MB, immediately revert and note "OOM risk" in description.
Reduce `max_seq_len` or `batch_size` if VRAM is tight.
