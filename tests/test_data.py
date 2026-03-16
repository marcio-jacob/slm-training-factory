"""
tests/test_data.py — Unit tests for data/registry.py and data/portuguese.py

Uses mocks for expensive HuggingFace dataset downloads so tests run offline/fast.
"""

from unittest.mock import patch

import pytest
from datasets import Dataset


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestDataRegistry:

    def test_register_and_load(self):
        from data.registry import _REGISTRY, register_dataset, load_dataset_by_name

        @register_dataset("_test-dataset")
        def _loader(**kwargs):
            return Dataset.from_dict({"text": ["hello", "world"]})

        ds = load_dataset_by_name("_test-dataset")
        assert len(ds) == 2
        assert "text" in ds.column_names

        # Cleanup
        del _REGISTRY["_test-dataset"]

    def test_unknown_dataset_raises(self):
        from data.registry import load_dataset_by_name
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_by_name("this-dataset-does-not-exist-xyz")

    def test_list_datasets_returns_sorted(self):
        from data.registry import list_datasets
        names = list_datasets()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert "wikipedia-pt" in names

    def test_kwargs_forwarded_to_loader(self):
        from data.registry import _REGISTRY, register_dataset, load_dataset_by_name

        received_kwargs = {}

        @register_dataset("_test-kwargs")
        def _loader(**kwargs):
            received_kwargs.update(kwargs)
            return Dataset.from_dict({"text": ["a"]})

        load_dataset_by_name("_test-kwargs", max_samples=5, split="train")
        assert received_kwargs.get("max_samples") == 5
        assert received_kwargs.get("split") == "train"

        del _REGISTRY["_test-kwargs"]


# ---------------------------------------------------------------------------
# Portuguese dataset tests (mocked HF downloads)
# ---------------------------------------------------------------------------

def _make_wikipedia_mock():
    """Returns a mock dataset resembling Wikipedia PT structure."""
    return Dataset.from_dict({
        "title": ["Brasil", "Portugal", "Angola"] * 5,
        "text": [
            "O Brasil é o maior país da América do Sul." * 5,
            "Portugal é um país ibérico com longa história." * 5,
            "Angola é um país africano com rica biodiversidade." * 5,
        ] * 5,
    })


class TestPortugueseDataset:

    @patch("data.portuguese.hf_load_dataset")
    def test_wikipedia_pt_returns_text_column(self, mock_hf):
        mock_hf.return_value = _make_wikipedia_mock()
        from data.portuguese import load_wikipedia_pt

        ds = load_wikipedia_pt(max_samples=5)
        assert "text" in ds.column_names
        assert len(ds) <= 5

    @patch("data.portuguese.hf_load_dataset")
    def test_wikipedia_pt_prepends_title(self, mock_hf):
        mock_hf.return_value = _make_wikipedia_mock()
        from data.portuguese import load_wikipedia_pt

        ds = load_wikipedia_pt(max_samples=3)
        # Title should be included in the text
        assert ds[0]["text"].startswith("#")

    @patch("data.portuguese.hf_load_dataset")
    def test_wikipedia_pt_filters_short(self, mock_hf):
        """Short articles (< 200 chars) should be filtered out."""
        ds_with_short = Dataset.from_dict({
            "title": ["Short", "Long"],
            "text": [
                "Too short.",
                "O Brasil é o maior país da América do Sul. " * 10,
            ],
        })
        mock_hf.return_value = ds_with_short
        from data.portuguese import load_wikipedia_pt

        ds = load_wikipedia_pt()
        # Short article should be filtered
        for item in ds:
            assert len(item["text"]) >= 200

    @patch("data.portuguese.hf_load_dataset")
    def test_portuguese_mix_combines_sources(self, mock_hf):
        """portuguese-mix should combine Wikipedia and mC4."""
        mock_ds = Dataset.from_dict({
            "title": ["A"] * 10,
            "text": ["Some long text content here. " * 10] * 10,
        })
        mock_hf.return_value = mock_ds
        from data.portuguese import load_portuguese_mix

        ds = load_portuguese_mix(max_samples=10)
        assert len(ds) > 0
        assert "text" in ds.column_names
