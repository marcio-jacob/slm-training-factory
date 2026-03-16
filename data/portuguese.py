"""
datasets/portuguese.py — Brazilian Portuguese text datasets for Stage 1 pretraining.

Registered datasets:
  - wikipedia-pt    : Portuguese Wikipedia (high quality, ~1GB, best starting point)
  - mc4-pt          : mC4 Portuguese (large, ~39GB, noisier)
  - oscar-pt        : OSCAR 2301 Portuguese (CC0, ~22GB)
  - portuguese-mix  : Wikipedia PT + mC4 PT combined (recommended for longer runs)

All loaders return a HuggingFace Dataset with a 'text' column.
"""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset as hf_load_dataset, concatenate_datasets, Dataset

from data.registry import register_dataset


def _filter_short(example, min_chars: int = 200) -> bool:
    """Drop articles shorter than min_chars — usually stubs or disambiguation pages."""
    return len(example.get("text", "")) >= min_chars


@register_dataset("wikipedia-pt")
def load_wikipedia_pt(
    max_samples: Optional[int] = None,
    split: str = "train",
    **kwargs,
) -> Dataset:
    """
    Portuguese Wikipedia via wikimedia/wikipedia.
    ~1GB, ~1M articles, CC-BY-SA license.
    High quality — great starting point for Portuguese foundation.
    """
    print("Loading Wikipedia PT (wikimedia/wikipedia 20231101.pt)...")
    ds = hf_load_dataset(
        "wikimedia/wikipedia",
        "20231101.pt",
        split=split,
        trust_remote_code=True,
    )

    # Merge title into text so the model learns article-style writing
    def _format(example):
        return {"text": f"# {example['title']}\n\n{example['text']}"}

    ds = ds.map(_format, remove_columns=[c for c in ds.column_names if c != "text"])
    ds = ds.filter(_filter_short)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"  Wikipedia PT: {len(ds):,} articles")
    return ds


@register_dataset("mc4-pt")
def load_mc4_pt(
    max_samples: Optional[int] = None,
    split: str = "train",
    **kwargs,
) -> Dataset:
    """
    mC4 Portuguese — multilingual C4, Portuguese slice.
    ~39GB streamed from HuggingFace. Use max_samples to cap for experiments.
    ODC-BY license.
    """
    print("Loading mC4 PT (allenai/c4 pt)...")
    # Stream to avoid downloading the full 39GB dataset
    ds = hf_load_dataset(
        "allenai/c4",
        "pt",
        split=split,
        streaming=False,
        trust_remote_code=True,
    )

    # mC4 already has a 'text' column
    ds = ds.filter(_filter_short)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"  mC4 PT: {len(ds):,} documents")
    return ds


@register_dataset("oscar-pt")
def load_oscar_pt(
    max_samples: Optional[int] = None,
    split: str = "train",
    **kwargs,
) -> Dataset:
    """
    OSCAR 2301 Portuguese — large web corpus, CC0 license.
    ~22GB, good complement to Wikipedia for informal language.
    """
    print("Loading OSCAR PT (oscar-corpus/OSCAR-2301 pt)...")
    ds = hf_load_dataset(
        "oscar-corpus/OSCAR-2301",
        language="pt",
        split=split,
        trust_remote_code=True,
    )

    # OSCAR uses 'content' column — rename to 'text'
    if "content" in ds.column_names:
        ds = ds.rename_column("content", "text")

    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    ds = ds.filter(_filter_short)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"  OSCAR PT: {len(ds):,} documents")
    return ds


@register_dataset("portuguese-mix")
def load_portuguese_mix(
    max_samples: Optional[int] = None,
    wikipedia_weight: float = 0.5,
    **kwargs,
) -> Dataset:
    """
    Wikipedia PT + mC4 PT combined.
    Wikipedia provides high-quality encyclopedic text;
    mC4 provides volume and informal/conversational coverage.

    Args:
        wikipedia_weight: Fraction of samples drawn from Wikipedia (0–1).
                          The rest comes from mC4.
        max_samples: Total samples across both sources.
    """
    total = max_samples or 200_000   # Default to 200K for a reasonable experiment

    n_wiki = int(total * wikipedia_weight)
    n_mc4 = total - n_wiki

    wiki = load_wikipedia_pt(max_samples=n_wiki)
    mc4 = load_mc4_pt(max_samples=n_mc4)

    combined = concatenate_datasets([wiki, mc4]).shuffle(seed=42)
    print(f"  Portuguese mix: {len(combined):,} total documents")
    return combined
