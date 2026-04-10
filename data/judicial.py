"""
data/judicial.py — Brazilian judicial corpus for legal domain adaptation.

Registered datasets:
  - judicial-br  : CF/88 + CC + CDC + CLT + CPC + CP + CPP + CTN + ECA + LGPD + LINDB

Sources are parquet files from the AttorneyCopilot silver layer:
  /home/spike/Desktop/AttorneyCopilot/attorney-copilot/data/silver/

Text is formatted as coherent legal documents (article-level granularity)
for causal language model continuation training.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from datasets import Dataset

from data.registry import register_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SILVER = Path("/home/spike/Desktop/AttorneyCopilot/attorney-copilot/data/silver")

_LAW_FILES = {
    "CC":    _SILVER / "laws" / "law__cc.parquet",
    "CDC":   _SILVER / "laws" / "law__cdc.parquet",
    "CLT":   _SILVER / "laws" / "law__clt.parquet",
    "CPC":   _SILVER / "laws" / "law__cpc.parquet",
    "CP":    _SILVER / "laws" / "law__cp.parquet",
    "CPP":   _SILVER / "laws" / "law__cpp.parquet",
    "CTN":   _SILVER / "laws" / "law__ctn.parquet",
    "ECA":   _SILVER / "laws" / "law__eca.parquet",
    "LGPD":  _SILVER / "laws" / "law__lgpd.parquet",
    "LINDB": _SILVER / "laws" / "law__lindb.parquet",
}

_CF_FILE = _SILVER / "cf" / "cf_provisions.parquet"

# ---------------------------------------------------------------------------
# Encoding fix
# ---------------------------------------------------------------------------
# The law parquets have ISO-8859-2 / CP1250 mojibake from the HTML parser:
# byte 0xE3 was decoded as iso-8859-2 → 'ă' instead of latin-1 → 'ã', etc.

_MOJIBAKE = str.maketrans({
    "ă": "ã",  "Ă": "Ã",
    "ŕ": "à",  "Ŕ": "À",
    "ę": "ê",  "Ę": "Ê",
    "ş": "º",              # nş → nº (ordinal masculine)
    "ő": "õ",  "Ő": "Õ",
})


def _fix(text: str) -> str:
    return text.translate(_MOJIBAKE)


# ---------------------------------------------------------------------------
# Noise patterns to strip from law raw_text
# ---------------------------------------------------------------------------

_NOISE = re.compile(
    r"\s*\("
    r"(?:Redaç[aã]o dada|Incluído|Incluída|Revogado|Revogada|Vigência|Vide|NR)"
    r"[^)]*\)",
    re.IGNORECASE,
)

_HEADER_TAGS = {"text", "td"}          # block_index metadata rows
_JUNK_PATTERN = re.compile(            # navigation / index noise
    r"^(ÍNDICE|Vigência|NR|L\d+compilada|Presidên|Casa Civil|Subchefia)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _fix(text)
    text = _NOISE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# CF/88 loader
# ---------------------------------------------------------------------------

def _load_cf() -> list[str]:
    """
    Load Constituição Federal provisions.

    Groups rows by artigo number so each training document contains a full
    article with all its paragraphs, incisos, and alíneas.
    Returns a list of formatted text strings.
    """
    import pandas as pd

    df = pd.read_parquet(_CF_FILE)
    documents: list[str] = []

    law_header = "Constituição da República Federativa do Brasil de 1988"

    # Rows with no artigo are section/title headers — skip them (they'll be
    # included as context in the first artigo of each section if needed).
    artigos = df[df["artigo"].notna() & (df["artigo"] != "NA")].copy()
    artigos["artigo"] = pd.to_numeric(artigos["artigo"], errors="coerce")
    artigos = artigos[artigos["artigo"].notna()]
    artigos["artigo"] = artigos["artigo"].astype(int)

    for artigo_num, group in artigos.groupby("artigo", sort=True):
        lines: list[str] = [law_header]

        # Include the titulo context if present
        titulos = group["titulo"].dropna().unique()
        if len(titulos):
            # Look up the título heading text
            titulo_rows = df[
                (df["titulo"].isin(titulos)) &
                (df["classe"] == "titartb") &
                (df["artigo"].isna())
            ]
            if not titulo_rows.empty:
                lines.append(titulo_rows.iloc[0]["texto"].strip())

        lines.append("")  # blank line before article

        for _, row in group.iterrows():
            text = str(row["texto"]).strip()
            if not text or text == "nan":
                continue
            lines.append(text)

        doc = "\n".join(lines).strip()
        if len(doc) >= 120:
            documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Law (codes) loader
# ---------------------------------------------------------------------------

def _load_law(key: str, path: Path) -> list[str]:
    """
    Load a Brazilian law code from parquet.

    Groups consecutive p-tagged rows into article-level documents using
    "Art." boundaries as delimiters.  Each document is prefixed with the
    law's full name and number.
    """
    import pandas as pd

    df = pd.read_parquet(path)

    # Extract law metadata from first row
    law_name = str(df.iloc[0]["law_name"])
    law_number = str(df.iloc[0]["law_number"])
    header = f"{law_name} ({law_number})"

    # Keep only paragraph-level content, drop header/navigation blocks
    content = df[~df["html_tag"].isin(_HEADER_TAGS)].copy()
    content = content[~content["raw_text"].str.match(_JUNK_PATTERN, na=False)]
    content = content.sort_values("block_index").reset_index(drop=True)

    documents: list[str] = []
    current_lines: list[str] = []

    def _flush():
        if current_lines:
            doc = header + "\n\n" + "\n".join(current_lines).strip()
            if len(doc) >= 150:
                documents.append(doc)
        current_lines.clear()

    for _, row in content.iterrows():
        raw = str(row.get("raw_text", "")).strip()
        if not raw or raw == "nan":
            continue

        text = _clean(raw)
        if not text:
            continue

        # New article — flush previous article into a document
        if re.match(r"^Art\.\s*\d", text):
            _flush()

        current_lines.append(text)

    _flush()
    return documents


# ---------------------------------------------------------------------------
# Registered dataset
# ---------------------------------------------------------------------------

@register_dataset("judicial-br")
def load_judicial_br(
    max_samples: Optional[int] = None,
    **kwargs,
) -> Dataset:
    """
    Brazilian judicial corpus: CF/88 + 10 major codes.

    Sources (all from AttorneyCopilot silver layer):
      CF/88, CC, CDC, CLT, CPC, CP, CPP, CTN, ECA, LGPD, LINDB

    Each training example is a self-contained legal provision (article + its
    paragraphs / incisos / alíneas) prefixed with the law name and number.
    Total: ~8,000–12,000 article-level documents.
    """
    documents: list[str] = []

    print("Loading CF/88…")
    cf_docs = _load_cf()
    print(f"  CF/88: {len(cf_docs):,} articles")
    documents.extend(cf_docs)

    for key, path in _LAW_FILES.items():
        if not path.exists():
            print(f"  [SKIP] {key}: file not found at {path}")
            continue
        law_docs = _load_law(key, path)
        print(f"  {key}: {len(law_docs):,} articles")
        documents.extend(law_docs)

    print(f"  Total judicial-br: {len(documents):,} documents")

    if max_samples is not None:
        documents = documents[:max_samples]

    return Dataset.from_dict({"text": documents})
