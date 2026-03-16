"""
metrics/perplexity.py — Language modeling perplexity evaluator.

Perplexity = exp(average cross-entropy loss).
Lower is better. Used as the primary metric for Stage 1 Portuguese pretraining.

This metric is model-agnostic — works with any HuggingFace CausalLM.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss


def evaluate_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_seq_len: int = 1024,
    batch_size: int = 2,
    device: Optional[str] = None,
) -> float:
    """
    Compute perplexity of a model on a list of text strings.

    Args:
        model: HuggingFace CausalLM (possibly quantized with PEFT adapters)
        tokenizer: Corresponding tokenizer
        texts: List of text strings to evaluate on
        max_seq_len: Truncate inputs to this length
        batch_size: Evaluation batch size
        device: Target device (defaults to model's device)

    Returns:
        Perplexity (float, lower is better)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    loss_fn = CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=True,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, T, vocab_size)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        B, T, V = shift_logits.shape
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)

        loss = loss_fn(flat_logits, flat_labels)

        # Count real (non-padded) tokens
        n_tokens = shift_mask.sum().item()
        # Recompute loss only on non-padded tokens
        masked_loss = (
            CrossEntropyLoss(reduction="none")(flat_logits, flat_labels)
            * shift_mask.view(-1).float()
        ).sum()

        total_loss += masked_loss.item()
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_perplexity_on_dataset(
    model,
    tokenizer,
    dataset,
    text_column: str = "text",
    max_samples: int = 500,
    max_seq_len: int = 1024,
    batch_size: int = 2,
) -> float:
    """
    Convenience wrapper: evaluate perplexity on a HuggingFace Dataset.

    Args:
        dataset: HuggingFace Dataset with a text column
        text_column: Name of the text column
        max_samples: Number of samples to evaluate on (for speed)
        ...rest: forwarded to evaluate_perplexity
    """
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    texts = dataset[text_column]
    return evaluate_perplexity(
        model, tokenizer, texts,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )
