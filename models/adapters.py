"""
models/adapters.py — LoRA / QLoRA adapter construction via PEFT.

Wraps a loaded (possibly quantized) model with trainable LoRA adapters.
Only the adapter parameters are trained, keeping the base model frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoraConfig:
    r: int = 64
    alpha: int = 128
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_dict(cls, d: dict) -> "LoraConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def apply_lora(model, lora_cfg: LoraConfig):
    """
    Wrap a model with LoRA adapters and return the PEFT model.

    The base model weights are frozen; only LoRA parameters are trainable.
    For quantized (4-bit) models, call models.registry.load_model() with
    load_in_4bit=True first — it calls prepare_model_for_kbit_training()
    which is required before this function.

    Returns:
        peft_model: PeftModel with trainable LoRA adapters attached
    """
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

    task_type = getattr(TaskType, lora_cfg.task_type)

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        task_type=task_type,
    )

    peft_model = get_peft_model(model, peft_config)

    trainable, total = peft_model.get_nb_trainable_parameters()
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"LoRA adapters applied:")
    print(f"  Trainable parameters: {trainable:,} ({pct:.2f}% of total {total:,})")
    print(f"  LoRA rank r={lora_cfg.r}, alpha={lora_cfg.alpha}")
    print(f"  Target modules: {lora_cfg.target_modules}")

    return peft_model


def load_adapter_for_inference(base_model, adapter_path: str):
    """
    Load a saved LoRA adapter onto an already-loaded base model for inference.
    Does NOT merge — the adapter remains separate (lower VRAM at inference time).
    """
    from peft import PeftModel
    print(f"Loading adapter for inference: {adapter_path}")
    return PeftModel.from_pretrained(base_model, adapter_path)
