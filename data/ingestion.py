"""
data/ingestion.py — Generic file ingestion for training data.

Supports:
  - Plain text (.txt, .md)
  - CSV (.csv) — reads a configurable text column
  - JSON / JSONL (.json, .jsonl) — reads a configurable text field
  - PDF (.pdf) — extracts text via pypdf (falls back to pdfminer if available)

All loaders return a HuggingFace Dataset with a single "text" column so they
plug directly into the training pipeline.

Usage:
    from data.ingestion import load_files, load_folder

    # Single file
    ds = load_files(["path/to/document.pdf"])

    # Whole folder (recursive)
    ds = load_folder("training_data/languages/pt-BR/")

    # CSV with custom column name
    ds = load_files(["data.csv"], csv_text_column="content")
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset

logger = logging.getLogger(__name__)

# Minimum character length for a text chunk to be kept
MIN_TEXT_LENGTH = 50

# File extensions handled by each loader
_TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
_CSV_EXTENSIONS = {".csv", ".tsv"}
_JSON_EXTENSIONS = {".json", ".jsonl"}
_PDF_EXTENSIONS = {".pdf"}

SUPPORTED_EXTENSIONS = _TEXT_EXTENSIONS | _CSV_EXTENSIONS | _JSON_EXTENSIONS | _PDF_EXTENSIONS


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> List[str]:
    """Read a plain-text or markdown file. Returns one entry per file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) >= MIN_TEXT_LENGTH:
            return [text]
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
    return []


def _read_csv_file(path: Path, text_column: Optional[str] = None) -> List[str]:
    """
    Read a CSV/TSV and extract the text column.

    Column selection order:
      1. `text_column` if provided
      2. First column named 'text', 'content', 'body', or 'document'
      3. First column in the file
    """
    delimiter = "\t" if path.suffix == ".tsv" else ","
    texts = []
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            if reader.fieldnames is None:
                return []

            # Resolve which column to use
            col = None
            if text_column and text_column in reader.fieldnames:
                col = text_column
            else:
                for candidate in ("text", "content", "body", "document"):
                    if candidate in reader.fieldnames:
                        col = candidate
                        break
                if col is None:
                    col = reader.fieldnames[0]

            for row in reader:
                val = (row.get(col) or "").strip()
                if len(val) >= MIN_TEXT_LENGTH:
                    texts.append(val)
    except Exception as e:
        logger.warning(f"Could not read CSV {path}: {e}")
    return texts


def _read_json_file(path: Path, text_field: Optional[str] = None) -> List[str]:
    """
    Read JSON or JSONL files.

    For JSONL: each line is a JSON object — extracts `text_field` (or auto-detects).
    For JSON:
      - If root is a list of strings → use directly
      - If root is a list of objects → extract text_field (or auto-detect)
      - If root is a single object with a 'text' key → wrap in list
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return []

    texts = []
    is_jsonl = path.suffix == ".jsonl" or "\n{" in raw[:200]

    def _extract_field(obj: dict) -> Optional[str]:
        if text_field and text_field in obj:
            return str(obj[text_field])
        for candidate in ("text", "content", "body", "document", "input", "output"):
            if candidate in obj:
                return str(obj[candidate])
        # Fallback: join all string values
        parts = [str(v) for v in obj.values() if isinstance(v, str)]
        return " ".join(parts) if parts else None

    if is_jsonl:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    val = obj
                elif isinstance(obj, dict):
                    val = _extract_field(obj) or ""
                else:
                    continue
                if len(val) >= MIN_TEXT_LENGTH:
                    texts.append(val)
            except json.JSONDecodeError:
                continue
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {path}: {e}")
            return []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    val = item
                elif isinstance(item, dict):
                    val = _extract_field(item) or ""
                else:
                    continue
                if len(val) >= MIN_TEXT_LENGTH:
                    texts.append(val)
        elif isinstance(data, dict):
            val = _extract_field(data) or ""
            if len(val) >= MIN_TEXT_LENGTH:
                texts.append(val)

    return texts


def _read_pdf_file(path: Path) -> List[str]:
    """
    Extract text from a PDF file.

    Tries pypdf first (pure Python, zero C deps).
    Falls back to pdfminer.six if pypdf is unavailable.
    Returns one text entry per page that has enough content.
    """
    # Try pypdf
    try:
        import pypdf  # noqa: F401
        reader = pypdf.PdfReader(str(path))
        texts = []
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if len(page_text) >= MIN_TEXT_LENGTH:
                texts.append(page_text)
        if texts:
            return texts
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pypdf failed on {path}: {e}")

    # Fall back to pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(path)).strip()
        if len(text) >= MIN_TEXT_LENGTH:
            return [text]
    except ImportError:
        logger.warning(
            f"No PDF library found to read {path}. "
            "Install one: `pip install pypdf` or `pip install pdfminer.six`"
        )
    except Exception as e:
        logger.warning(f"pdfminer failed on {path}: {e}")

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_files(
    paths: List[Union[str, Path]],
    csv_text_column: Optional[str] = None,
    json_text_field: Optional[str] = None,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> Dataset:
    """
    Load text from a list of files (mixed formats supported).

    Args:
        paths: File paths to load. Unsupported extensions are skipped with a warning.
        csv_text_column: Column name to use for CSV files (auto-detected if None).
        json_text_field: Field name to use for JSON/JSONL files (auto-detected if None).
        min_text_length: Minimum characters for a text chunk to be kept.

    Returns:
        HuggingFace Dataset with a single "text" column.
    """
    texts: List[str] = []

    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        ext = path.suffix.lower()

        if ext in _TEXT_EXTENSIONS:
            chunks = _read_text_file(path)
        elif ext in _CSV_EXTENSIONS:
            chunks = _read_csv_file(path, text_column=csv_text_column)
        elif ext in _JSON_EXTENSIONS:
            chunks = _read_json_file(path, text_field=json_text_field)
        elif ext in _PDF_EXTENSIONS:
            chunks = _read_pdf_file(path)
        else:
            logger.warning(f"Unsupported file extension '{ext}' for {path} — skipping")
            continue

        # Apply caller's min_text_length if different from default
        if min_text_length != MIN_TEXT_LENGTH:
            chunks = [c for c in chunks if len(c) >= min_text_length]

        texts.extend(chunks)
        logger.info(f"Loaded {len(chunks)} chunks from {path}")

    if not texts:
        logger.warning("No text chunks loaded — returning empty dataset")

    return Dataset.from_dict({"text": texts})


def load_folder(
    folder: Union[str, Path],
    recursive: bool = True,
    csv_text_column: Optional[str] = None,
    json_text_field: Optional[str] = None,
    min_text_length: int = MIN_TEXT_LENGTH,
    extensions: Optional[set] = None,
) -> Dataset:
    """
    Load all supported files from a folder.

    Args:
        folder: Path to directory to scan.
        recursive: If True, descend into sub-directories.
        csv_text_column: Column name for CSV files.
        json_text_field: Field name for JSON/JSONL files.
        min_text_length: Minimum characters per chunk.
        extensions: Restrict to these extensions (e.g. {".pdf", ".txt"}).
                    Defaults to all SUPPORTED_EXTENSIONS.

    Returns:
        HuggingFace Dataset with a single "text" column.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    allowed = extensions if extensions is not None else SUPPORTED_EXTENSIONS
    glob_pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(glob_pattern) if p.is_file() and p.suffix.lower() in allowed]
    files.sort()

    logger.info(f"Found {len(files)} file(s) in {folder}")

    return load_files(
        files,
        csv_text_column=csv_text_column,
        json_text_field=json_text_field,
        min_text_length=min_text_length,
    )
