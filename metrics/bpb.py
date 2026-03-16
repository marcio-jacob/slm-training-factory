"""
metrics/bpb.py — Bits-per-byte (BPB) metric for HuggingFace CausalLMs.

BPB = cross_entropy_loss / ln(2) / avg_bytes_per_token

Adapted from the original prepare.py evaluate_bpb() function.
BPB is vocabulary-size independent, making it a fair comparison metric
even when switching models with different tokenizers.

Lower is better (like perplexity, but normalized for tokenizer efficiency).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss


def _avg_bytes_per_token(tokenizer, sample_texts: list[str], n_samples: int = 100) -> float:
    """Estimate average bytes per token for this tokenizer on these texts."""
    total_bytes = 0
    total_tokens = 0
    for text in sample_texts[:n_samples]:
        tokens = tokenizer.encode(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokens)
    if total_tokens == 0:
        return 1.0
    return total_bytes / total_tokens


def evaluate_bpb(
    model,
    tokenizer,
    texts: list[str],
    max_seq_len: int = 1024,
    batch_size: int = 2,
    device: Optional[str] = None,
) -> float:
    """
    Compute bits-per-byte on a list of text strings.

    Args:
        model: HuggingFace CausalLM
        tokenizer: Corresponding tokenizer
        texts: Evaluation text strings
        max_seq_len: Truncation length
        batch_size: Eval batch size
        device: Target device

    Returns:
        BPB (float, lower is better)
    """
    if device is None:
        device = next(model.parameters()).device

    avg_bytes_per_tok = _avg_bytes_per_token(tokenizer, texts)

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        B, T, V = shift_logits.shape
        per_token_loss = CrossEntropyLoss(reduction="none")(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
        ).view(B, T)

        masked_loss = (per_token_loss * shift_mask).sum()
        n_tokens = shift_mask.sum()

        total_loss += masked_loss.item()
        total_tokens += n_tokens.item()

    if total_tokens == 0:
        return float("inf")

    avg_nats_per_token = total_loss / total_tokens
    nats_per_byte = avg_nats_per_token / avg_bytes_per_tok
    bpb = nats_per_byte / math.log(2)
    return bpb


def evaluate_bpb_on_dataset(
    model,
    tokenizer,
    dataset,
    text_column: str = "text",
    max_samples: int = 500,
    max_seq_len: int = 1024,
    batch_size: int = 2,
) -> float:
    """Evaluate BPB on a HuggingFace Dataset."""
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    texts = dataset[text_column]
    return evaluate_bpb(model, tokenizer, texts, max_seq_len=max_seq_len, batch_size=batch_size)
