"""
tests/test_trainers.py — Unit tests for trainers/base.py and trainers/qlora.py

Tests the TrainingConfig, TrainingResult dataclasses and the TimeBudgetCallback.
Full QLoRA trainer integration is covered in test_e2e.py.
"""

import time

import pytest


class TestTrainingConfig:

    def test_from_dict_basic(self):
        from trainers.base import TrainingConfig
        cfg = TrainingConfig.from_dict({
            "batch_size": 4,
            "learning_rate": 1e-4,
            "budget_seconds": 7200,
        })
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 1e-4
        assert cfg.budget_seconds == 7200

    def test_from_dict_ignores_unknown_keys(self):
        from trainers.base import TrainingConfig
        cfg = TrainingConfig.from_dict({"batch_size": 2, "nonexistent": "ignored"})
        assert cfg.batch_size == 2

    def test_defaults(self):
        from trainers.base import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.gradient_checkpointing is True
        assert cfg.bf16 is True
        assert cfg.fp16 is False


class TestTrainingResult:

    def test_as_dict_contains_required_keys(self):
        from trainers.base import TrainingResult
        r = TrainingResult(
            metric_name="perplexity",
            metric_value=42.5,
            training_seconds=100.0,
            total_seconds=120.0,
            peak_vram_mb=4096.0,
            num_steps=50,
            checkpoint_path="/tmp/ckpt",
        )
        d = r.as_dict()
        assert d["metric_name"] == "perplexity"
        assert d["metric_value"] == 42.5
        assert d["training_seconds"] == 100.0
        assert d["num_steps"] == 50

    def test_str_representation(self):
        from trainers.base import TrainingResult
        r = TrainingResult(
            metric_name="perplexity",
            metric_value=42.5,
            training_seconds=100.0,
            total_seconds=120.0,
            peak_vram_mb=4096.0,
            num_steps=50,
        )
        s = str(r)
        assert "42.5" in s
        assert "perplexity" in s

    def test_extra_dict_included(self):
        from trainers.base import TrainingResult
        r = TrainingResult(
            metric_name="bpb",
            metric_value=1.5,
            training_seconds=60.0,
            total_seconds=80.0,
            peak_vram_mb=2048.0,
            num_steps=10,
            extra={"eval_loss": 3.2},
        )
        d = r.as_dict()
        assert d["eval_loss"] == 3.2


class TestTimeBudgetCallback:

    def test_stops_training_after_budget(self):
        from trainers.qlora import TimeBudgetCallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        cb = TimeBudgetCallback(budget_seconds=1)
        state = TrainerState()
        control = TrainerControl()
        args = TrainingArguments(output_dir="/tmp", use_cpu=True)

        # Simulate training begin
        cb.on_train_begin(args, state, control)

        # Wait for budget to expire
        time.sleep(1.1)

        # Simulate step end
        cb.on_step_end(args, state, control)
        assert control.should_training_stop is True

    def test_does_not_stop_before_budget(self):
        from trainers.qlora import TimeBudgetCallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        cb = TimeBudgetCallback(budget_seconds=3600)
        state = TrainerState()
        control = TrainerControl()
        args = TrainingArguments(output_dir="/tmp", use_cpu=True)

        cb.on_train_begin(args, state, control)
        cb.on_step_end(args, state, control)

        assert control.should_training_stop is not True

    def test_callback_without_train_begin_does_not_crash(self):
        """on_step_end before on_train_begin should be a no-op."""
        from trainers.qlora import TimeBudgetCallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        cb = TimeBudgetCallback(budget_seconds=1)
        state = TrainerState()
        control = TrainerControl()
        args = TrainingArguments(output_dir="/tmp", use_cpu=True)

        # Do NOT call on_train_begin first
        cb.on_step_end(args, state, control)
        # Should not crash, should_training_stop not set
        assert control.should_training_stop is not True
