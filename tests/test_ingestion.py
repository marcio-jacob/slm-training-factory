"""
tests/test_ingestion.py — Unit tests for data/ingestion.py

Tests all file-format loaders without hitting the network.
"""

import csv
import json
import textwrap
from pathlib import Path

import pytest

from data.ingestion import (
    load_files,
    load_folder,
    _read_text_file,
    _read_csv_file,
    _read_json_file,
    _read_pdf_file,
    SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Text file reader
# ---------------------------------------------------------------------------

class TestTextReader:

    def test_reads_plain_text(self, tmp_path):
        p = _write(tmp_path, "doc.txt", "Hello world! " * 10)
        chunks = _read_text_file(p)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_reads_markdown(self, tmp_path):
        p = _write(tmp_path, "notes.md", "# Title\n\nSome text here. " * 5)
        chunks = _read_text_file(p)
        assert len(chunks) == 1

    def test_skips_short_content(self, tmp_path):
        p = _write(tmp_path, "tiny.txt", "Hi")
        chunks = _read_text_file(p)
        assert chunks == []


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

class TestCsvReader:

    def _make_csv(self, tmp_path, rows, header):
        p = tmp_path / "data.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        return p

    def test_reads_text_column(self, tmp_path):
        p = self._make_csv(tmp_path,
            [{"text": "A long enough sentence about something important. " * 3}],
            ["text"])
        chunks = _read_csv_file(p)
        assert len(chunks) == 1
        assert "important" in chunks[0]

    def test_auto_detects_content_column(self, tmp_path):
        p = self._make_csv(tmp_path,
            [{"id": "1", "content": "Important legal text here. " * 5}],
            ["id", "content"])
        chunks = _read_csv_file(p)
        assert len(chunks) == 1

    def test_explicit_column_override(self, tmp_path):
        p = self._make_csv(tmp_path,
            [{"body": "Legal corpus text. " * 5, "other": "ignore me"}],
            ["body", "other"])
        chunks = _read_csv_file(p, text_column="body")
        assert "Legal" in chunks[0]

    def test_skips_short_rows(self, tmp_path):
        p = self._make_csv(tmp_path,
            [{"text": "short"}, {"text": "Long enough text here. " * 5}],
            ["text"])
        chunks = _read_csv_file(p)
        assert len(chunks) == 1

    def test_multiple_rows(self, tmp_path):
        long = "This is a sufficiently long line of text. " * 3
        p = self._make_csv(tmp_path,
            [{"text": long}, {"text": long}, {"text": long}],
            ["text"])
        chunks = _read_csv_file(p)
        assert len(chunks) == 3


# ---------------------------------------------------------------------------
# JSON / JSONL reader
# ---------------------------------------------------------------------------

class TestJsonReader:

    def test_reads_list_of_strings(self, tmp_path):
        long = "A long piece of text that should definitely be included. " * 3
        p = _write(tmp_path, "d.json", json.dumps([long, long]))
        chunks = _read_json_file(p)
        assert len(chunks) == 2

    def test_reads_list_of_objects(self, tmp_path):
        long = "Sufficient content for the filter. " * 4
        data = [{"text": long}, {"text": long}]
        p = _write(tmp_path, "d.json", json.dumps(data))
        chunks = _read_json_file(p)
        assert len(chunks) == 2

    def test_reads_jsonl(self, tmp_path):
        long = "Enough text to pass the length filter. " * 3
        lines = "\n".join(json.dumps({"text": long}) for _ in range(3))
        p = _write(tmp_path, "d.jsonl", lines)
        chunks = _read_json_file(p)
        assert len(chunks) == 3

    def test_explicit_field(self, tmp_path):
        long = "Legal document content sufficient for training. " * 3
        data = [{"body": long, "meta": "ignore"}]
        p = _write(tmp_path, "d.json", json.dumps(data))
        chunks = _read_json_file(p, text_field="body")
        assert "Legal" in chunks[0]

    def test_auto_detects_content_field(self, tmp_path):
        long = "Content field auto-detected here. " * 4
        data = [{"content": long}]
        p = _write(tmp_path, "d.json", json.dumps(data))
        chunks = _read_json_file(p)
        assert len(chunks) == 1

    def test_invalid_json_returns_empty(self, tmp_path):
        p = _write(tmp_path, "bad.json", "NOT JSON{{{")
        chunks = _read_json_file(p)
        assert chunks == []

    def test_input_output_pairs_in_jsonl(self, tmp_path):
        """Instruction-tuning pairs with 'output' field should be picked up."""
        long_answer = (
            "Water is essential for life and covers approximately 71% of Earth's surface. "
            "It is vital for all known forms of life on our planet."
        )
        lines = json.dumps({"input": "What is water?", "output": long_answer})
        p = _write(tmp_path, "qa.jsonl", lines)
        # Extract the long "output" field (input is too short to pass the length filter)
        chunks = _read_json_file(p, text_field="output")
        assert len(chunks) == 1
        assert "71%" in chunks[0]


# ---------------------------------------------------------------------------
# load_files (integration)
# ---------------------------------------------------------------------------

class TestLoadFiles:

    def test_returns_dataset_with_text_column(self, tmp_path):
        long = "Dataset text column should be present. " * 4
        p = _write(tmp_path, "doc.txt", long)
        ds = load_files([p])
        assert "text" in ds.column_names
        assert len(ds) == 1

    def test_mixed_formats(self, tmp_path):
        long = "Enough text. " * 6
        txt = _write(tmp_path, "a.txt", long)
        json_file = _write(tmp_path, "b.json", json.dumps([long, long]))
        csv_file = tmp_path / "c.csv"
        with csv_file.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["text"])
            w.writeheader()
            w.writerow({"text": long})
        ds = load_files([txt, json_file, csv_file])
        assert len(ds) == 4  # 1 txt + 2 json + 1 csv

    def test_missing_file_skipped(self, tmp_path):
        ds = load_files([tmp_path / "nonexistent.txt"])
        assert len(ds) == 0

    def test_unsupported_extension_skipped(self, tmp_path):
        p = _write(tmp_path, "data.xyz", "content " * 20)
        ds = load_files([p])
        assert len(ds) == 0

    def test_custom_min_text_length(self, tmp_path):
        long = "This text is long enough to pass the default length filter. " * 2
        p = _write(tmp_path, "doc.txt", long)
        # With a very high min_text_length, the chunk is filtered out
        ds = load_files([p], min_text_length=9999)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# load_folder (integration)
# ---------------------------------------------------------------------------

class TestLoadFolder:

    def test_loads_all_files_in_folder(self, tmp_path):
        long = "Folder loading test content. " * 5
        _write(tmp_path, "a.txt", long)
        _write(tmp_path, "b.txt", long)
        ds = load_folder(tmp_path)
        assert len(ds) == 2

    def test_recursive_subfolder(self, tmp_path):
        long = "Recursive loading test. " * 5
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(tmp_path, "root.txt", long)
        _write(sub, "nested.txt", long)
        ds = load_folder(tmp_path, recursive=True)
        assert len(ds) == 2

    def test_non_recursive_ignores_subfolders(self, tmp_path):
        long = "Non-recursive test. " * 5
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(tmp_path, "root.txt", long)
        _write(sub, "nested.txt", long)
        ds = load_folder(tmp_path, recursive=False)
        assert len(ds) == 1

    def test_extension_filter(self, tmp_path):
        long = "Extension filter test. " * 5
        _write(tmp_path, "a.txt", long)
        _write(tmp_path, "b.json", json.dumps([long]))
        ds = load_folder(tmp_path, extensions={".txt"})
        assert len(ds) == 1

    def test_raises_on_nonexistent_folder(self):
        with pytest.raises(ValueError, match="Not a directory"):
            load_folder("/nonexistent/path/xyz")

    def test_empty_folder_returns_empty_dataset(self, tmp_path):
        ds = load_folder(tmp_path)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# PDF reader (mocked — no real PDF needed)
# ---------------------------------------------------------------------------

class TestPdfReader:

    def test_returns_empty_when_no_pdf_library(self, tmp_path, monkeypatch):
        """When neither pypdf nor pdfminer is available, return empty list."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("pypdf", "pdfminer", "pdfminer.high_level"):
                raise ImportError(f"Mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        p = tmp_path / "fake.pdf"
        p.write_bytes(b"%PDF-1.4 fake content")
        chunks = _read_pdf_file(p)
        assert chunks == []
