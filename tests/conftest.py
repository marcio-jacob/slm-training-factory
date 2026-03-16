"""
tests/conftest.py — Shared fixtures for all test modules.

Uses 'sshleifer/tiny-gpt2' (a 2-layer GPT-2 with 10M params) as the test model
so tests run without downloading large checkpoints and without requiring GPU.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Tiny model used in all tests — no GPU required, instant download (<10MB)
TEST_MODEL_NAME = "sshleifer/tiny-gpt2"

# Disable tokenizers parallelism warnings in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer():
    """Load the tiny test model once per test session."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)
    model.eval()
    return model, tokenizer


@pytest.fixture
def tmp_store(tmp_path):
    """A temporary ModelStore directory."""
    return str(tmp_path / "test_store")


@pytest.fixture
def sample_texts():
    """Small list of Portuguese and legal texts for metric tests."""
    return [
        "O Brasil é um país de dimensões continentais com grande diversidade cultural.",
        "A Constituição Federal de 1988 estabelece os direitos fundamentais dos cidadãos.",
        "O Superior Tribunal de Justiça julgou procedente o recurso especial interposto.",
        "A legislação brasileira prevê penas para crimes contra o patrimônio público.",
        "Wikipedia é uma enciclopédia multilíngue de conteúdo livre e colaborativo.",
    ] * 4  # 20 samples total
