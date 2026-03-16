"""
trainers/qlora.py — QLoRA trainer targeting 6GB VRAM.

Uses HuggingFace Trainer with:
- 4-bit quantized base model (loaded by models/registry.py)
- LoRA adapters (applied by models/adapters.py)
- Gradient checkpointing + bfloat16 compute
- Time-budget stopping callback (stops after config.budget_seconds)
- Saves only LoRA adapter weights (not the full model)

Effective batch size = batch_size × gradient_accumulation_steps × max_seq_len tokens
Default: 2 × 8 = 16 sequences × 1024 tokens = 16K tokens/step
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from trainers.base import BaseTrainer, TrainingConfig, TrainingResult


def _best_available_optimizer() -> str:
    """Use 8-bit paged AdamW when bitsandbytes is available (saves VRAM),
    otherwise fall back to standard AdamW (for CPU / test environments)."""
    try:
        import bitsandbytes  # noqa: F401
        return "paged_adamw_8bit"
    except ImportError:
        return "adamw_torch"


class TimeBudgetCallback(TrainerCallback):
    """Stop training after budget_seconds of wall-clock time."""

    def __init__(self, budget_seconds: int):
        self.budget_seconds = budget_seconds
        self.start_time: Optional[float] = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if elapsed >= self.budget_seconds:
            print(f"\nTime budget reached ({elapsed:.0f}s >= {self.budget_seconds}s). Stopping.")
            control.should_training_stop = True


class QLoRATrainer(BaseTrainer):
    """
    QLoRA trainer: trains only the LoRA adapter parameters on a 4-bit quantized model.

    The model passed in should already have LoRA adapters applied
    (via models/adapters.py apply_lora). This trainer handles tokenization,
    data collation, the training loop, and saving checkpoints.
    """

    def _tokenize_dataset(self, dataset, max_seq_len: int):
        """Tokenize and chunk the dataset into fixed-length sequences."""

        def _tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_len,
                padding=False,
            )

        # num_proc=None avoids any subprocess/pool creation — runs in-process.
        # Using num_proc>=1 creates a process pool even for 1 worker, which
        # cannot inherit the CUDA context and crashes when CUDA is initialized.
        tokenized = dataset.map(
            _tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        return tokenized

    def train(self) -> TrainingResult:
        t_total_start = time.time()
        cfg = self.config

        print(f"\n=== QLoRA Training ===")
        print(f"Budget: {cfg.budget_seconds}s | LR: {cfg.learning_rate} | "
              f"Batch: {cfg.batch_size}×{cfg.gradient_accumulation_steps} | "
              f"SeqLen: {cfg.max_seq_len}")

        # Tokenize datasets
        print("Tokenizing training dataset...")
        train_tok = self._tokenize_dataset(self.train_dataset, cfg.max_seq_len)
        print("Tokenizing eval dataset...")
        eval_tok = self._tokenize_dataset(self.eval_dataset, cfg.max_seq_len)

        # Data collator — standard causal LM (no masking)
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            bf16=cfg.bf16,
            fp16=cfg.fp16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            # Eval & saving
            # Eval runs every save_steps; if budget ends before first eval,
            # we run eval manually after training so load_best is safe.
            eval_strategy="steps",
            eval_steps=cfg.save_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=2,          # Keep only best 2 checkpoints
            load_best_model_at_end=False, # We run final eval ourselves
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Logging
            logging_steps=cfg.logging_steps,
            report_to="none",            # No wandb/tensorboard by default
            # Misc
            dataloader_num_workers=2,
            remove_unused_columns=False,
            # Max steps: very large — time budget callback stops us
            max_steps=999_999,
            optim=_best_available_optimizer(),  # paged_adamw_8bit when bnb available
        )

        t_train_start = time.time()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=collator,
            callbacks=[TimeBudgetCallback(cfg.budget_seconds)],
        )

        train_result = trainer.train()

        t_train_end = time.time()

        # Save LoRA adapter only (not the full model — much smaller)
        print(f"\nSaving LoRA adapter to {cfg.output_dir} ...")
        self.model.save_pretrained(cfg.output_dir)
        self.tokenizer.save_pretrained(cfg.output_dir)

        # VRAM stats
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        # Final eval
        eval_loss = trainer.evaluate()["eval_loss"]
        perplexity = float(torch.exp(torch.tensor(eval_loss)).item())

        t_total_end = time.time()

        result = TrainingResult(
            metric_name="perplexity",
            metric_value=perplexity,
            training_seconds=t_train_end - t_train_start,
            total_seconds=t_total_end - t_total_start,
            peak_vram_mb=peak_vram_mb,
            num_steps=train_result.global_step,
            checkpoint_path=cfg.output_dir,
            extra={"eval_loss": eval_loss},
        )

        print(f"\n{result}")
        return result

    def evaluate(self) -> float:
        """Standalone evaluation — returns perplexity on eval_dataset."""
        cfg = self.config
        eval_tok = self._tokenize_dataset(self.eval_dataset, cfg.max_seq_len)
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_eval_batch_size=cfg.batch_size,
            bf16=cfg.bf16,
            fp16=cfg.fp16,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_tok,
            data_collator=collator,
        )
        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        return float(torch.exp(torch.tensor(eval_loss)).item())
