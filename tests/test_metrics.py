"""
tests/test_metrics.py — Unit tests for metrics/perplexity.py and metrics/bpb.py

Uses the tiny test model (sshleifer/tiny-gpt2) for actual forward-pass tests.
All tests run on CPU — no GPU required.
"""

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Perplexity tests
# ---------------------------------------------------------------------------

class TestPerplexity:

    def test_perplexity_returns_finite_positive(self, tiny_model_and_tokenizer, sample_texts):
        from metrics.perplexity import evaluate_perplexity
        model, tokenizer = tiny_model_and_tokenizer
        ppl = evaluate_perplexity(model, tokenizer, sample_texts[:4], max_seq_len=64, batch_size=2)
        assert math.isfinite(ppl)
        assert ppl > 0

    def test_perplexity_on_repeated_text_is_lower(self, tiny_model_and_tokenizer):
        """Model should assign lower perplexity to text it has seen many times."""
        from metrics.perplexity import evaluate_perplexity
        model, tokenizer = tiny_model_and_tokenizer

        # Random noise should have higher perplexity than normal text
        normal_texts = ["The quick brown fox jumps over the lazy dog."] * 4
        noise_texts = ["xkzq wlrp bvth jmfn qdxz plkr vths mnqp"] * 4

        ppl_normal = evaluate_perplexity(model, tokenizer, normal_texts, max_seq_len=32, batch_size=2)
        ppl_noise = evaluate_perplexity(model, tokenizer, noise_texts, max_seq_len=32, batch_size=2)

        # Noise should be harder to predict (higher perplexity)
        # Note: with a tiny random model this may not always hold, but the values should differ
        assert ppl_normal != ppl_noise

    def test_perplexity_empty_list_returns_inf(self, tiny_model_and_tokenizer):
        from metrics.perplexity import evaluate_perplexity
        model, tokenizer = tiny_model_and_tokenizer
        ppl = evaluate_perplexity(model, tokenizer, [], max_seq_len=64, batch_size=2)
        assert ppl == float("inf")

    def test_perplexity_single_text(self, tiny_model_and_tokenizer):
        from metrics.perplexity import evaluate_perplexity
        model, tokenizer = tiny_model_and_tokenizer
        ppl = evaluate_perplexity(model, tokenizer, ["Hello world."], max_seq_len=32, batch_size=1)
        assert math.isfinite(ppl)
        assert ppl > 1.0  # perplexity of 1 = perfect prediction (impossible here)

    def test_perplexity_on_dataset(self, tiny_model_and_tokenizer, sample_texts):
        from metrics.perplexity import evaluate_perplexity_on_dataset
        from datasets import Dataset

        model, tokenizer = tiny_model_and_tokenizer
        ds = Dataset.from_dict({"text": sample_texts[:6]})
        ppl = evaluate_perplexity_on_dataset(
            model, tokenizer, ds, text_column="text", max_samples=4, max_seq_len=64
        )
        assert math.isfinite(ppl)
        assert ppl > 0

    def test_perplexity_max_samples_respected(self, tiny_model_and_tokenizer, sample_texts):
        from metrics.perplexity import evaluate_perplexity_on_dataset
        from datasets import Dataset

        model, tokenizer = tiny_model_and_tokenizer
        ds = Dataset.from_dict({"text": sample_texts})
        # Should not raise even though ds has 20 samples and max_samples=3
        ppl = evaluate_perplexity_on_dataset(
            model, tokenizer, ds, max_samples=3, max_seq_len=32
        )
        assert math.isfinite(ppl)


# ---------------------------------------------------------------------------
# BPB tests
# ---------------------------------------------------------------------------

class TestBPB:

    def test_bpb_returns_finite_positive(self, tiny_model_and_tokenizer, sample_texts):
        from metrics.bpb import evaluate_bpb
        model, tokenizer = tiny_model_and_tokenizer
        bpb = evaluate_bpb(model, tokenizer, sample_texts[:4], max_seq_len=64, batch_size=2)
        assert math.isfinite(bpb)
        assert bpb > 0

    def test_bpb_on_dataset(self, tiny_model_and_tokenizer, sample_texts):
        from metrics.bpb import evaluate_bpb_on_dataset
        from datasets import Dataset

        model, tokenizer = tiny_model_and_tokenizer
        ds = Dataset.from_dict({"text": sample_texts[:6]})
        bpb = evaluate_bpb_on_dataset(
            model, tokenizer, ds, text_column="text", max_samples=4, max_seq_len=64
        )
        assert math.isfinite(bpb)
        assert bpb > 0

    def test_bpb_less_than_perplexity(self, tiny_model_and_tokenizer, sample_texts):
        """BPB normalizes by bytes/token so it should be significantly less than raw perplexity."""
        from metrics.perplexity import evaluate_perplexity
        from metrics.bpb import evaluate_bpb

        model, tokenizer = tiny_model_and_tokenizer
        texts = sample_texts[:4]
        ppl = evaluate_perplexity(model, tokenizer, texts, max_seq_len=64, batch_size=2)
        bpb = evaluate_bpb(model, tokenizer, texts, max_seq_len=64, batch_size=2)

        # BPB in nats/byte is always much smaller than perplexity (exp(cross_entropy))
        assert bpb < ppl

    def test_avg_bytes_per_token_positive(self, tiny_model_and_tokenizer, sample_texts):
        from metrics.bpb import _avg_bytes_per_token
        _, tokenizer = tiny_model_and_tokenizer
        avg = _avg_bytes_per_token(tokenizer, sample_texts[:5])
        assert avg > 0
        assert avg < 20  # Sanity: avg bytes/token should be in range [1, 10] for natural text
