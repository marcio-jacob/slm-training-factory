# Training Data

This directory holds raw training corpora for the SLM factory. It is **gitignored** — add your own files here without worrying about accidentally committing large datasets.

```
training_data/
└── languages/          ← language foundation training (Phase 2)
    ├── pt-BR/          ← Brazilian Portuguese
    ├── en-US/          ← English
    └── <lang-code>/    ← Add any language using ISO 639-1 + region code
```

## Supported file formats

The ingestion pipeline (`data/ingestion.py`) accepts:

| Format | Extensions | Notes |
|--------|-----------|-------|
| Plain text | `.txt`, `.md`, `.rst` | One file → one chunk |
| CSV / TSV | `.csv`, `.tsv` | Auto-detects text column |
| JSON | `.json` | List of strings, list of objects, or single object |
| JSONL | `.jsonl` | One JSON object per line |
| PDF | `.pdf` | Requires `pypdf` (`pip install pypdf`) |

## Loading data

```python
from data.ingestion import load_files, load_folder

# Load everything from a language folder
ds = load_folder("training_data/languages/pt-BR/")

# Load specific files
ds = load_files([
    "training_data/languages/pt-BR/wikipedia_articles.jsonl",
    "training_data/languages/pt-BR/news_corpus.csv",
])
```

## Adding a language corpus

Place files in `languages/<lang-code>/`. The factory accepts any language — just point your config at the folder:

```yaml
dataset:
  path: training_data/languages/fr-FR/
```

Or use a built-in HuggingFace loader (no local files needed):

```yaml
dataset:
  name: wikipedia-pt   # streams directly from HuggingFace
```

## Parsing raw text into training format

Use `data_parser.py` to convert raw text into structured `{"input": "...", "output": "..."}` pairs for instruction-tuning:

```bash
python data_parser.py \
  --input  training_data/languages/pt-BR/raw_corpus.txt \
  --output training_data/languages/pt-BR/parsed.jsonl \
  --domain "Brazilian Portuguese text"
```

Requires `ANTHROPIC_API_KEY` in `.env`.
