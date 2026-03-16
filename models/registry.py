"""
models/registry.py — HuggingFace model loader with optional 4-bit QLoRA quantization.

Supports any AutoModelForCausalLM-compatible model from HuggingFace Hub or a local path.
For 6GB VRAM, always use load_in_4bit=True with a 7B model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelConfig:
    name: str
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _build_bnb_config(cfg: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.load_in_4bit:
        return None
    dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def load_model(cfg: ModelConfig, device_map: str = "auto"):
    """
    Load a causal LM and its tokenizer from HuggingFace or a local path.

    Returns:
        model: AutoModelForCausalLM (quantized if load_in_4bit=True)
        tokenizer: AutoTokenizer with pad_token set
    """
    bnb_config = _build_bnb_config(cfg)

    print(f"Loading model: {cfg.name}")
    if cfg.load_in_4bit:
        print("  Quantization: 4-bit NF4 (QLoRA mode)")

    # local_files_only avoids a slow/hanging network check when the model
    # is already in the HuggingFace cache (~/.cache/huggingface/hub/)
    _local = Path(cfg.name).exists()  # True for explicit local paths
    _local_files_only = _local or bool(os.environ.get("TRANSFORMERS_OFFLINE"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=_local_files_only,
    )
    # Many causal LMs don't have a pad token — use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if cfg.load_in_4bit:
        # Workaround for transformers 5.x + bitsandbytes on low-VRAM GPUs:
        # The new core_model_loading.py materializes tensors in bf16 directly
        # to GPU, OOMing before quantization can happen. Loading to CPU first
        # (quantizes in bf16 on CPU RAM) then .cuda() moves the 4-bit tensors
        # to GPU — uses only ~0.5 bytes/param on the GPU.
        model = AutoModelForCausalLM.from_pretrained(
            cfg.name,
            quantization_config=bnb_config,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=_local_files_only,
        )
        model = model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=_local_files_only,
        )

    # Required before adding LoRA adapters to a quantized model
    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.1f}M")

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"  VRAM after load: {vram_mb:.0f} MB")

    return model, tokenizer


def estimate_vram_mb(model_name: str, load_in_4bit: bool = True) -> float:
    """
    Rough VRAM estimate (MB) without actually loading the model.
    Useful for --dry-run checks.
    """
    # Heuristic: parameter count × bytes per param × overhead factor
    # Known sizes for common models
    _known = {
        "Qwen/Qwen1.5-7B": 7_700_000_000,
        "Qwen/Qwen1.5-4B": 4_000_000_000,
        "Qwen/Qwen1.5-1.8B": 1_800_000_000,
        "Qwen/Qwen2-7B": 7_600_000_000,
        "Qwen/Qwen2.5-1.5B": 1_500_000_000,
        "Qwen/Qwen2.5-3B": 3_100_000_000,
        "Qwen/Qwen2.5-7B": 7_600_000_000,
        "Qwen/Qwen2.5-7B-Instruct": 7_600_000_000,
        "Qwen/Qwen3-8B": 8_200_000_000,
    }
    n_params = _known.get(model_name, 7_000_000_000)

    if load_in_4bit:
        bytes_per_param = 0.5  # 4-bit = 0.5 bytes
        overhead = 1.3         # LoRA adapters + activations
    else:
        bytes_per_param = 2.0  # bfloat16
        overhead = 1.2

    return n_params * bytes_per_param * overhead / 1024**2


def merge_adapter_into_base(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    device: str = "cpu",
):
    """
    Merge a trained LoRA adapter back into the base model weights and save
    a standalone model ready for Stage 2 fine-tuning or inference.

    Usage:
        python factory.py merge --base Qwen/Qwen2.5-7B \
            --adapter ./checkpoints/qwen-portuguese-best \
            --output ./models/qwen-portuguese-v1
    """
    from peft import PeftModel

    print(f"Loading base model in bfloat16 for merge: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")
