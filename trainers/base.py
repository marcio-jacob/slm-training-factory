"""
trainers/base.py — Abstract trainer interface.

All trainer implementations (QLoRA, full fine-tune, etc.) must subclass BaseTrainer
and implement train() and evaluate(). This lets factory.py swap trainers
without changing any other code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 1024
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    budget_seconds: int = 3600
    save_steps: int = 200
    logging_steps: int = 10
    output_dir: str = "./checkpoints"

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingResult:
    """Returned by trainer.train() — logged to results TSV by the autoresearch loop."""
    metric_name: str
    metric_value: float
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    num_steps: int
    checkpoint_path: Optional[str] = None
    extra: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "training_seconds": self.training_seconds,
            "total_seconds": self.total_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "num_steps": self.num_steps,
            "checkpoint_path": self.checkpoint_path or "",
        }
        if self.extra:
            d.update(self.extra)
        return d

    def __str__(self) -> str:
        return (
            f"{self.metric_name}:          {self.metric_value:.6f}\n"
            f"training_seconds: {self.training_seconds:.1f}\n"
            f"total_seconds:    {self.total_seconds:.1f}\n"
            f"peak_vram_mb:     {self.peak_vram_mb:.1f}\n"
            f"num_steps:        {self.num_steps}\n"
            f"checkpoint:       {self.checkpoint_path or 'none'}"
        )


class BaseTrainer(ABC):
    """
    Abstract trainer. Subclasses implement train() and evaluate().
    """

    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

    @abstractmethod
    def train(self) -> TrainingResult:
        """
        Run training for up to config.budget_seconds wall-clock seconds.
        Returns a TrainingResult with the primary metric and timing info.
        """
        ...

    @abstractmethod
    def evaluate(self) -> float:
        """
        Run evaluation on eval_dataset.
        Returns the primary metric value (lower is better for perplexity/BPB).
        """
        ...
