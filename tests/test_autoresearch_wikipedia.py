"""
tests/test_autoresearch_wikipedia.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
First autoresearch iteration: Wikipedia Portuguese → perplexity validation.

What this test does:
  1.  Stream ~120 real articles from Wikipedia PT (wikimedia/wikipedia)
  2.  Split into 100 train / 20 validation articles
  3.  Load sshleifer/tiny-gpt2 and measure BASELINE perplexity on the PT val set
  4.  Apply LoRA adapters (rank 8) to the model
  5.  Fine-tune for up to 60 seconds on the PT training articles
  6.  Re-measure POST-TRAINING perplexity on the SAME PT val set
  7.  Try to promote to the model version store (only saved if metric improved)
  8.  Print a detailed optimization report

Why tiny-gpt2 and not Qwen?
  tiny-gpt2 (~10 MB) lets this test run without a large download or GPU.
  The pipeline is identical — swap the model name in config to use Qwen 7B
  for a real production run.

Expected result:
  Perplexity DECREASES after training on Portuguese text.
  Even a randomly initialized tiny model will reduce its loss on the training
  domain after a few gradient steps — this is the fundamental signal that
  the autoresearch loop relies on to compare configs.

Run:
    python -m pytest tests/test_autoresearch_wikipedia.py -v -s
"""

import math
import sys
import time
from pathlib import Path

import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "sshleifer/tiny-gpt2"
TRAIN_BUDGET = 60          # seconds of LoRA fine-tuning
N_TRAIN      = 100         # Wikipedia PT articles for training
N_VAL        = 20          # Wikipedia PT articles for validation
LORA_RANK    = 8
LORA_ALPHA   = 16
LEARNING_RATE = 5e-3       # High enough to see clear movement in 60 s on CPU
MAX_SEQ_LEN  = 128         # Short for speed; real runs use 1024+
BATCH_SIZE   = 2
GRAD_ACCUM   = 2           # Smaller accum = more frequent updates in budget


# ── Wikipedia PT loader (real data, streamed) ──────────────────────────────────

def _load_wikipedia_pt_streaming(n_total: int):
    """
    Stream the first n_total articles from Wikipedia PT.
    Uses `streaming=True` so we never download the full dataset.
    Returns a list of dicts: [{"title": ..., "text": ...}, ...]
    """
    print(f"  Streaming {n_total} articles from wikimedia/wikipedia (pt)...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.pt",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    samples = []
    for item in ds.take(n_total):
        text = f"# {item['title']}\n\n{item['text']}"
        # Keep only articles with enough content
        if len(text) >= 300:
            samples.append(text)
        if len(samples) >= n_total:
            break
    print(f"  Got {len(samples)} articles (min 300 chars each)")
    return samples


def _measure_perplexity(model, tokenizer, texts, label=""):
    """Compute perplexity on a list of texts.
    Device is inferred from the model so this works on both CPU and GPU."""
    from metrics.perplexity import evaluate_perplexity
    model.eval()
    # Auto-detect model device (may be GPU after HuggingFace Trainer)
    device = next(model.parameters()).device
    ppl = evaluate_perplexity(
        model, tokenizer, texts,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        device=device,
    )
    if label:
        print(f"  {label}: {ppl:.4f}  (device={device})")
    return ppl


def _banner(msg):
    print(f"\n{'━'*62}")
    print(f"  {msg}")
    print(f"{'━'*62}")


# ── Main test ──────────────────────────────────────────────────────────────────

def test_autoresearch_wikipedia_portuguese(tmp_path):
    """
    Full autoresearch iteration on Wikipedia PT.

    Validates:
    - Real PT articles are loadable and usable for training
    - Baseline perplexity is measurable on PT val set
    - LoRA fine-tuning runs without error
    - Post-training perplexity is a finite positive number
    - Version store records the best checkpoint correctly
    - Version is only promoted when metric improves
    """
    _banner("Autoresearch — Wikipedia Portuguese — First Iteration")
    wall_start = time.time()

    # ── Step 1: Load real Wikipedia PT data ────────────────────────
    print(f"\n[1/7] Loading Wikipedia PT articles...")
    all_texts = _load_wikipedia_pt_streaming(n_total=N_TRAIN + N_VAL)

    # If streaming returns fewer articles than expected, adjust gracefully
    n_val   = max(5, min(N_VAL, len(all_texts) // 5))
    n_train = len(all_texts) - n_val

    train_texts = all_texts[:n_train]
    val_texts   = all_texts[n_train:]

    print(f"  Train: {len(train_texts)} articles")
    print(f"  Val:   {len(val_texts)} articles")
    assert len(val_texts) >= 5, "Need at least 5 validation articles"

    # ── Step 2: Load model ─────────────────────────────────────────
    print(f"\n[2/7] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Step 3: Baseline perplexity (pre-training) ─────────────────
    print(f"\n[3/7] Baseline perplexity (no training)...")
    baseline_ppl = _measure_perplexity(base_model, tokenizer, val_texts,
                                       label="Baseline PPL on PT val")
    assert math.isfinite(baseline_ppl) and baseline_ppl > 0, \
        f"Baseline PPL should be finite and positive, got {baseline_ppl}"

    # ── Step 4: Apply LoRA ─────────────────────────────────────────
    print(f"\n[4/7] Applying LoRA (r={LORA_RANK}, alpha={LORA_ALPHA})...")
    from models.adapters import LoraConfig, apply_lora

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        alpha=LORA_ALPHA,
        target_modules=["c_attn"],   # tiny-gpt2 attention projection
        dropout=0.0,
        task_type="CAUSAL_LM",
    )
    model = apply_lora(base_model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} "
          f"({100.0 * trainable / n_params:.2f}% of total)")

    # ── Step 5: Fine-tune on Wikipedia PT ─────────────────────────
    print(f"\n[5/7] Fine-tuning on {len(train_texts)} Wikipedia PT articles "
          f"(budget={TRAIN_BUDGET}s)...")

    from datasets import Dataset as HFDataset
    from trainers.base import TrainingConfig
    from trainers.qlora import QLoRATrainer

    store_dir  = str(tmp_path / "versions")
    output_dir = str(tmp_path / "checkpoint")

    train_ds = HFDataset.from_dict({"text": train_texts})
    val_ds   = HFDataset.from_dict({"text": val_texts})

    train_cfg = TrainingConfig(
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_seq_len=MAX_SEQ_LEN,
        learning_rate=LEARNING_RATE,
        lr_scheduler="constant",        # constant LR — no warmup schedule issues
        warmup_ratio=0.0,               # no warmup: LR is full from step 1
        weight_decay=0.0,
        gradient_checkpointing=False,   # not needed for tiny CPU model
        bf16=False,
        fp16=False,
        budget_seconds=TRAIN_BUDGET,
        save_steps=9999,                # don't checkpoint mid-run (speed)
        logging_steps=5,
        output_dir=output_dir,
    )

    t_train = time.time()
    trainer = QLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        config=train_cfg,
    )
    train_result = trainer.train()
    t_train_elapsed = time.time() - t_train

    print(f"  Training time:  {t_train_elapsed:.1f}s")
    print(f"  Steps taken:    {train_result.num_steps}")
    assert train_result.num_steps >= 1, \
        f"Expected at least 1 training step, got {train_result.num_steps}"

    # ── Step 6: Post-training evaluation ──────────────────────────
    print(f"\n[6/7] Post-training evaluation on PT val set...")
    model.eval()
    post_ppl = _measure_perplexity(model, tokenizer, val_texts,
                                   label="Post-train PPL on PT val")
    assert math.isfinite(post_ppl) and post_ppl > 0, \
        f"Post-training PPL should be finite and positive, got {post_ppl}"

    # ── Step 7: Version store ──────────────────────────────────────
    print(f"\n[7/7] Version store — promote if improved...")
    from versioning.model_store import ModelStore

    store = ModelStore(store_dir, lower_is_better=True)
    store.init(base_model_name=MODEL_NAME, metric_name="perplexity-pt")

    # Register baseline first (as if this were the initial state)
    r_baseline = store.try_promote(
        model=model, tokenizer=tokenizer,
        metric_value=baseline_ppl,
        config={"lora": {"r": 0}, "note": "baseline, no training"},
        description="Baseline (no LoRA, no training)",
    )

    # Register post-training result
    r_trained = store.try_promote(
        model=model, tokenizer=tokenizer,
        metric_value=post_ppl,
        config={
            "lora": {"r": LORA_RANK, "alpha": LORA_ALPHA},
            "training": {"learning_rate": LEARNING_RATE,
                         "budget_seconds": TRAIN_BUDGET,
                         "steps": train_result.num_steps},
            "dataset": "wikipedia-pt",
        },
        description=(
            f"LoRA r={LORA_RANK} on Wikipedia PT, "
            f"{train_result.num_steps} steps"
        ),
        parent_version=store.best_version,
    )

    # ── Print optimization report ──────────────────────────────────
    _banner("OPTIMIZATION REPORT — Wikipedia Portuguese")

    delta     = baseline_ppl - post_ppl
    pct       = 100.0 * delta / baseline_ppl if baseline_ppl > 0 else 0.0
    direction = "▼ IMPROVED" if delta > 0 else "▲ DEGRADED" if delta < 0 else "= UNCHANGED"

    print(f"""
  Model:              {MODEL_NAME}
  LoRA rank:          r={LORA_RANK}, alpha={LORA_ALPHA}
  Training data:      {len(train_texts)} Wikipedia PT articles
  Validation data:    {len(val_texts)} Wikipedia PT articles
  Steps:              {train_result.num_steps}
  Training time:      {t_train_elapsed:.1f}s
  Total wall time:    {time.time() - wall_start:.1f}s

  Baseline PPL:       {baseline_ppl:>12.4f}
  Post-train PPL:     {post_ppl:>12.4f}
  Δ Perplexity:       {delta:>+12.4f}  ({pct:+.2f}%)  {direction}

  Baseline promoted:  {r_baseline}  (always yes — first entry)
  Trained promoted:   {r_trained}   (yes iff PPL improved)
  Best version:       {store.best_version}
  Best PPL:           {store.best_metric:.4f}
""")

    store.print_summary()
    print(f"\n{'━'*62}\n")

    # ── Assertions ─────────────────────────────────────────────────
    # 1. Baseline always gets saved (first entry)
    assert r_baseline is True, "Baseline must always be promoted as the first version"

    # 2. Training ran
    assert train_result.num_steps >= 1

    # 3. Post-training PPL is finite
    assert math.isfinite(post_ppl) and post_ppl > 0

    # 4. Store has at least one version on disk
    assert store.best_version is not None
    assert Path(store.get_adapter_path(store.best_version)).exists()

    # 5. The best version stores the lowest PPL
    versions = store.list_versions()
    assert versions[0]["metric_value"] == store.best_metric

    # 6. If PPL improved, trained version was promoted and is the best
    if r_trained:
        assert store.best_metric == post_ppl, \
            "If post-train was promoted it should be the new best"
        print(f"  ✓ Model IMPROVED: PPL {baseline_ppl:.2f} → {post_ppl:.2f}"
              f"  ({pct:+.2f}%)")
    else:
        print(f"  ~ No improvement this run (PPL {baseline_ppl:.2f} → "
              f"{post_ppl:.2f}), best stays at {store.best_metric:.2f}")
        print(f"    This is expected for a random model on very few steps.")

    # 7. Version metadata is correct
    best_meta = store.get_best()
    assert best_meta["metric_name"] == "perplexity-pt"
    assert best_meta["base_model"] == MODEL_NAME
