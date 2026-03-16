"""
tests/test_models.py — Unit tests for models/registry.py and models/adapters.py

Uses the tiny test model to verify loading, adapter application, and VRAM estimation.
Does NOT test 4-bit quantization (requires specific GPU + bitsandbytes install).
"""

import pytest


class TestModelConfig:

    def test_from_dict_basic(self):
        from models.registry import ModelConfig
        cfg = ModelConfig.from_dict({
            "name": "test/model",
            "load_in_4bit": False,
            "bnb_4bit_compute_dtype": "float16",
        })
        assert cfg.name == "test/model"
        assert cfg.load_in_4bit is False
        assert cfg.bnb_4bit_compute_dtype == "float16"

    def test_from_dict_ignores_unknown_keys(self):
        from models.registry import ModelConfig
        cfg = ModelConfig.from_dict({
            "name": "test/model",
            "unknown_key": "should be ignored",
        })
        assert cfg.name == "test/model"

    def test_defaults(self):
        from models.registry import ModelConfig
        cfg = ModelConfig(name="test/model")
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True


class TestEstimateVram:

    def test_known_model_estimate(self):
        from models.registry import estimate_vram_mb
        est = estimate_vram_mb("Qwen/Qwen1.5-7B", load_in_4bit=True)
        # 7B × 0.5 bytes × 1.3 overhead / (1024^2) ≈ 4300 MB
        assert 3000 < est < 7000

    def test_4bit_lower_than_full_precision(self):
        from models.registry import estimate_vram_mb
        est_4bit = estimate_vram_mb("Qwen/Qwen1.5-7B", load_in_4bit=True)
        est_full = estimate_vram_mb("Qwen/Qwen1.5-7B", load_in_4bit=False)
        assert est_4bit < est_full

    def test_unknown_model_defaults_to_7b_estimate(self):
        from models.registry import estimate_vram_mb
        est = estimate_vram_mb("unknown/model-xyz", load_in_4bit=True)
        assert est > 0


class TestLoadModelWithoutQuantization:
    """Load tiny-gpt2 without 4-bit quantization (no bitsandbytes needed)."""

    def test_load_model_returns_model_and_tokenizer(self):
        from models.registry import ModelConfig, load_model
        cfg = ModelConfig(
            name="sshleifer/tiny-gpt2",
            load_in_4bit=False,  # No quantization for CPU test
        )
        model, tokenizer = load_model(cfg, device_map="cpu")
        assert model is not None
        assert tokenizer is not None

    def test_tokenizer_has_pad_token(self):
        from models.registry import ModelConfig, load_model
        cfg = ModelConfig(name="sshleifer/tiny-gpt2", load_in_4bit=False)
        _, tokenizer = load_model(cfg, device_map="cpu")
        assert tokenizer.pad_token is not None


class TestLoraConfig:

    def test_from_dict(self):
        from models.adapters import LoraConfig
        cfg = LoraConfig.from_dict({
            "r": 32,
            "alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "dropout": 0.1,
        })
        assert cfg.r == 32
        assert cfg.alpha == 64
        assert cfg.target_modules == ["q_proj", "v_proj"]
        assert cfg.dropout == 0.1

    def test_defaults(self):
        from models.adapters import LoraConfig
        cfg = LoraConfig()
        assert cfg.r == 64
        assert cfg.alpha == 128
        assert "q_proj" in cfg.target_modules

    def test_from_dict_ignores_unknown(self):
        from models.adapters import LoraConfig
        cfg = LoraConfig.from_dict({"r": 16, "unknown": "ignored"})
        assert cfg.r == 16


class TestApplyLora:

    def test_apply_lora_increases_trainable_params(self):
        """After LoRA, some parameters should be trainable that weren't before."""
        from transformers import AutoModelForCausalLM
        from models.adapters import LoraConfig, apply_lora

        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

        # tiny-gpt2 uses c_attn / c_proj — use those as LoRA targets
        lora_cfg = LoraConfig(
            r=4,
            alpha=8,
            target_modules=["c_attn"],
            dropout=0.0,
            task_type="CAUSAL_LM",
        )
        peft_model = apply_lora(base, lora_cfg)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())

        # Some (but not all) params should be trainable
        assert 0 < trainable < total

    def test_apply_lora_preserves_forward(self, tiny_model_and_tokenizer):
        """LoRA model should still produce logits for input tokens."""
        import torch
        from models.adapters import LoraConfig, apply_lora

        base, tokenizer = tiny_model_and_tokenizer
        lora_cfg = LoraConfig(
            r=4, alpha=8,
            target_modules=["c_attn"],
            dropout=0.0,
            task_type="CAUSAL_LM",
        )
        peft_model = apply_lora(base, lora_cfg)

        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            out = peft_model(**inputs)
        assert out.logits is not None
        assert out.logits.shape[-1] > 0  # vocab size
