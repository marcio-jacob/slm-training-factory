"""
gen_compare.py — Portuguese text generation comparison: v0001 vs v0004.

Tests raw generation quality (not Q&A) with open-ended continuation prompts.
Each model runs in its own subprocess to avoid CUDA memory issues.

Usage:
    python gen_compare.py
    python gen_compare.py --worker=v0001 --out=/tmp/gen_v0001.json
    python gen_compare.py --worker=v0004 --out=/tmp/gen_v0004.json
    python gen_compare.py --display-only
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL  = "Qwen/Qwen2.5-3B"
V0001_PATH  = "./models/qwen-portuguese-v0001"
V0004_PATH  = "./models/qwen-portuguese-phase3-versions/v0004"

RESULTS_V0001 = "/tmp/gen_v0001.json"
RESULTS_V0004 = "/tmp/gen_v0004.json"

W = 82

# ---------------------------------------------------------------------------
# Generation prompts — pure text continuation, no instructions.
# Designed to stress raw Portuguese fluency and coherence.
# ---------------------------------------------------------------------------
PROMPTS = [
    {
        "category": "Narrativa literária",
        "prefix": (
            "O velho pescador acordou antes do amanhecer, como fazia todos os dias "
            "há quarenta anos. O cheiro do mar entrou pela janela entreaberta e ele "
            "fechou os olhos por um instante,"
        ),
    },
    {
        "category": "Texto jornalístico",
        "prefix": (
            "A taxa de desemprego no Brasil recuou para 7,8% no segundo trimestre, "
            "o menor índice desde 2014, segundo dados divulgados pelo IBGE nesta "
            "quinta-feira. O resultado superou as expectativas dos analistas,"
        ),
    },
    {
        "category": "Texto acadêmico / científico",
        "prefix": (
            "A relação entre o desmatamento da Amazônia e as mudanças nos padrões "
            "de precipitação no centro-sul do Brasil tem sido objeto de crescente "
            "debate científico. Estudos recentes indicam que a remoção da cobertura "
            "vegetal compromete"
        ),
    },
    {
        "category": "Diálogo cotidiano",
        "prefix": (
            "— Você viu o jogo ontem à noite? — perguntou Rodrigo, puxando a cadeira "
            "e se sentando à mesa do bar.\n"
            "— Vi sim. Que decepção — respondeu Paulo, balançando a cabeça. — "
            "Perdemos de um jeito que"
        ),
    },
    {
        "category": "Texto histórico",
        "prefix": (
            "A Proclamação da República, ocorrida em 15 de novembro de 1889, marcou "
            "o fim do Segundo Reinado e o início de uma nova era política no Brasil. "
            "O marechal Deodoro da Fonseca, à frente de tropas militares,"
        ),
    },
    {
        "category": "Texto filosófico / reflexivo",
        "prefix": (
            "A liberdade, conceito central na tradição filosófica ocidental, assume "
            "significados distintos conforme o contexto em que é analisada. Para "
            "Jean-Paul Sartre, o ser humano está condenado a ser livre, pois"
        ),
    },
    {
        "category": "Receita / texto instrucional",
        "prefix": (
            "Para preparar um autêntico feijão tropeiro mineiro, comece separando "
            "os ingredientes: feijão carioca cozido, bacon, linguiça calabresa, "
            "farinha de mandioca, ovos, couve e alho. Numa frigideira grande,"
        ),
    },
    {
        "category": "Crônica urbana",
        "prefix": (
            "São Paulo, cidade que nunca dorme, mostrava mais uma vez o seu ritmo "
            "frenético. Às sete da manhã, a Avenida Paulista já estava tomada por "
            "uma multidão de rostos apressados. Entre eles, Maria da Conceição"
        ),
    },
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_adapter(adapter_path: str):
    abs_adapter = str(Path(adapter_path).resolve())
    print(f"  Loading base {BASE_MODEL} + adapter {adapter_path} …")
    tok = AutoTokenizer.from_pretrained(abs_adapter)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=_bnb(),
        device_map="cpu",
    )
    base = base.to("cuda:0")
    mdl = PeftModel.from_pretrained(base, abs_adapter)
    mdl.eval()
    return mdl, tok


# ---------------------------------------------------------------------------
# Generation — raw continuation (no system prompt, no chat template)
# ---------------------------------------------------------------------------
GEN_KWARGS = dict(
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.92,
    repetition_penalty=1.15,
    do_sample=True,
)


def generate_continuation(model, tokenizer, prefix: str) -> str:
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, **GEN_KWARGS)

    new_tokens = out[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_worker(adapter_path: str, out_path: str):
    model, tok = load_adapter(adapter_path)
    results = []
    for p in PROMPTS:
        cont = generate_continuation(model, tok, p["prefix"])
        results.append({
            "category": p["category"],
            "prefix":   p["prefix"],
            "continuation": cont,
        })
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out_path}")


def spawn(label: str, adapter_path: str, out_path: str):
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    print(f"\n{'─'*W}")
    print(f"  Running {label} ({adapter_path})")
    print(f"{'─'*W}")
    r = subprocess.run(
        [sys.executable, __file__, f"--worker={adapter_path}", f"--out={out_path}"],
        env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{label} worker failed")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def wrap(text: str, indent: int = 4) -> str:
    pref = " " * indent
    return textwrap.fill(
        text, width=W - indent, initial_indent=pref, subsequent_indent=pref
    )


def display(v0001: list, v0004: list):
    print("\n" + "═" * W)
    print("  PORTUGUESE GENERATION: v0001  vs  v0004")
    print("  (raw continuation — no instructions, no chat template)")
    print("═" * W)

    for a, b in zip(v0001, v0004):
        print(f"\n{'─'*W}")
        print(f"  CATEGORY: {a['category']}")
        print(f"{'─'*W}")
        print(f"\n  PREFIX:")
        print(wrap(a["prefix"]))

        print(f"\n  v0001 continuation:")
        print(wrap(a["continuation"]))

        print(f"\n  v0004 continuation:")
        print(wrap(b["continuation"]))

    print(f"\n{'═'*W}")
    print("  What to look for:")
    print("   • Grammatical correctness (agreement, verb tenses)")
    print("   • Lexical naturalness (idiomatic PT vocabulary)")
    print("   • Coherence with the prefix style/register")
    print("   • Absence of English words or non-PT tokens")
    print("═" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    spawn("v0001", V0001_PATH, RESULTS_V0001)
    spawn("v0004", V0004_PATH, RESULTS_V0004)

    with open(RESULTS_V0001) as f:
        v0001 = json.load(f)
    with open(RESULTS_V0004) as f:
        v0004 = json.load(f)

    display(v0001, v0004)


if __name__ == "__main__":
    worker_path = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--worker=")), None)
    worker_out  = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    display_only = "--display-only" in sys.argv

    if worker_path:
        run_worker(worker_path, worker_out)
    elif display_only:
        with open(RESULTS_V0001) as f: v0001 = json.load(f)
        with open(RESULTS_V0004) as f: v0004 = json.load(f)
        display(v0001, v0004)
    else:
        main()
