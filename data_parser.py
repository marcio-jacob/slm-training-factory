"""
data_parser.py — Convert raw text chunks into structured training pairs.

Uses Claude (via the Anthropic API) to parse raw text into
{"input": "...", "output": "..."} JSON pairs optimised for SLM instruction-tuning.

Usage:
    python data_parser.py \\
        --input  training_data/languages/pt-BR/raw/ \\
        --output training_data/languages/pt-BR/parsed_qa.jsonl \\
        --domain "Brazilian Portuguese text" \\
        --batch-size 10

Requirements:
    pip install anthropic python-dotenv

Environment:
    ANTHROPIC_API_KEY must be set — either in a .env file at the project root
    or as a shell environment variable.

Output format (JSONL):
    Each line is a JSON object with "input" and "output" keys, e.g.:
    {"input": "What is preventive detention in Brazil?", "output": "Preventive detention (prisão preventiva) is..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Load .env before importing anthropic so the key is visible
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — fall back to shell env

try:
    import anthropic
except ImportError:
    print(
        "ERROR: 'anthropic' package not found.\n"
        "Install it with:  pip install anthropic\n",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096
MIN_CHUNK_LEN = 100   # Skip chunks shorter than this (too thin to parse)
MAX_CHUNK_LEN = 8000  # Truncate very long chunks to stay within token budget

_SYSTEM_PROMPT = """\
You are an expert training-data curator for specialised language models.

Your task is to convert raw text chunks into high-quality question-answer pairs
that can be used to instruction-tune a small language model (SLM) on a specific domain.

Rules:
1. Generate exactly ONE {"input": "...", "output": "..."} JSON object per text chunk.
2. The "input" should be a realistic question a domain practitioner would ask.
3. The "output" should be a concise, factual, self-contained answer drawn from the text.
4. Do NOT include information not present in the text.
5. Write in the same language as the input text.
6. Return ONLY the JSON object — no explanation, no markdown, no extra text.

Example:
Text: "Water covers about 71% of Earth's surface and is essential for all known life."
Output: {"input": "What fraction of Earth's surface is covered by water?", "output": "Water covers about 71% of Earth's surface and is essential for all known life."}
"""


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def _build_user_message(chunk: str, domain: str) -> str:
    truncated = chunk[:MAX_CHUNK_LEN]
    domain_hint = f" (domain: {domain})" if domain else ""
    return (
        f"Convert the following text{domain_hint} into one training pair "
        f"as a single JSON object with 'input' and 'output' keys.\n\n"
        f"Text:\n{truncated}"
    )


def parse_chunk(
    client: anthropic.Anthropic,
    chunk: str,
    domain: str = "",
    retries: int = 3,
) -> Optional[dict]:
    """
    Ask Claude to produce one {"input": ..., "output": ...} pair from a text chunk.

    Returns a dict on success, None on failure.
    """
    for attempt in range(retries):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_message(chunk, domain)}],
            ) as stream:
                text = stream.get_final_message().content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            pair = json.loads(text)

            if "input" in pair and "output" in pair:
                return pair

            logger.warning("Claude returned JSON without 'input'/'output' keys: %s", text[:200])

        except json.JSONDecodeError as e:
            logger.warning("JSON decode error on attempt %d: %s", attempt + 1, e)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            logger.warning("Rate limited. Waiting %ds before retry…", wait)
            time.sleep(wait)
        except anthropic.APIError as e:
            logger.warning("API error on attempt %d: %s", attempt + 1, e)
            time.sleep(5)

    return None


def parse_chunks_batch(
    client: anthropic.Anthropic,
    chunks: List[str],
    domain: str = "",
    batch_size: int = 10,
) -> List[dict]:
    """
    Parse a list of text chunks into training pairs, processing batch_size at a time.
    Returns a list of successfully parsed pairs.
    """
    pairs = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        logger.info(
            "Processing chunks %d–%d of %d…",
            i + 1, min(i + len(batch), total), total
        )

        for j, chunk in enumerate(batch):
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_LEN:
                logger.debug("Skipping short chunk (%d chars)", len(chunk))
                continue

            pair = parse_chunk(client, chunk, domain)
            if pair:
                pairs.append(pair)
                logger.debug("  [%d/%d] ✓  %s", i + j + 1, total, pair["input"][:60])
            else:
                logger.warning("  [%d/%d] ✗  Failed to parse chunk", i + j + 1, total)

        # Brief pause between batches to be polite to the API
        if i + batch_size < total:
            time.sleep(1)

    return pairs


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_input_chunks(input_path: Path) -> List[str]:
    """Load raw text chunks from a file or folder using data/ingestion.py."""
    from data.ingestion import load_files, load_folder

    if input_path.is_dir():
        ds = load_folder(str(input_path))
    elif input_path.is_file():
        ds = load_files([str(input_path)])
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    return ds["text"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse raw text chunks into {input, output} training pairs using Claude.\n\n"
            "Requires ANTHROPIC_API_KEY in environment or .env file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a file or folder of raw text documents.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path for the output JSONL file of training pairs.",
    )
    parser.add_argument(
        "--domain", "-d", default="",
        help=(
            "Short description of the domain, e.g. 'Brazilian law'. "
            "Used as context for Claude when generating questions."
        ),
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=10,
        help="Number of chunks to process per batch (default: 10).",
    )
    parser.add_argument(
        "--max-chunks", "-m", type=int, default=None,
        help="Maximum number of chunks to process (default: all).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and count chunks without calling the API.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        logger.error(
            "ANTHROPIC_API_KEY not set.\n"
            "  • Add it to a .env file in the project root: ANTHROPIC_API_KEY=sk-ant-...\n"
            "  • Or export it in your shell before running this script."
        )
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load chunks
    logger.info("Loading text chunks from: %s", input_path)
    chunks = _load_input_chunks(input_path)

    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    logger.info("Loaded %d text chunk(s)", len(chunks))

    if args.dry_run:
        logger.info("Dry run — stopping before API calls.")
        for i, c in enumerate(chunks[:5], 1):
            print(f"\n[chunk {i}] {c[:120]}{'…' if len(c) > 120 else ''}")
        return

    if not chunks:
        logger.warning("No chunks loaded — nothing to parse.")
        sys.exit(0)

    # Parse
    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.time()

    logger.info(
        "Parsing with Claude %s (domain=%r, batch_size=%d)…",
        MODEL, args.domain, args.batch_size
    )

    pairs = parse_chunks_batch(
        client, chunks,
        domain=args.domain,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - t0
    logger.info(
        "Parsed %d/%d chunks → %d training pairs  (%.1fs)",
        len(chunks), len(chunks), len(pairs), elapsed
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Saved %d pairs to: %s", len(pairs), output_path)

    # Preview
    if pairs:
        print("\n── Sample output ─────────────────────────────────────────────")
        for pair in pairs[:3]:
            print(json.dumps(pair, ensure_ascii=False, indent=2))
        if len(pairs) > 3:
            print(f"  … and {len(pairs) - 3} more.")
        print("──────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
