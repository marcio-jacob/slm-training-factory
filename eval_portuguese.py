"""
Portuguese performance benchmark: Qwen2.5-3B (base) vs Qwen2.5-3B + PT adapter.

Evaluation axes:
  1. Perplexity  — on held-out Portuguese reference sentences (lower = better)
  2. Qualitative — side-by-side completions on tricky Portuguese prompts
  3. Summary     — per-category win/loss table printed at the end
"""

import gc
import json
import math
import os
import subprocess
import sys
import textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ADAPTER_PATH = "./models/qwen-portuguese-v0001"
BASE_MODEL   = "Qwen/Qwen2.5-3B"

# ---------------------------------------------------------------------------
# Tricky Portuguese prompts  (category → list of prompts)
# Designed to stress areas where fine-tuning on PT wikipedia should help:
#   - subjunctive mood, crase, complex conjugation, idioms, formal register,
#     cultural knowledge, orthographic challenges, ambiguity resolution.
# ---------------------------------------------------------------------------
EVAL_PROMPTS = {
    "Gramática – Subjuntivo": [
        "Complete a frase corretamente: 'Espero que ele ______ (vir) amanhã.'",
        "Corrija se necessário: 'É importante que todos estudem bastante.' Explique o motivo.",
        "Qual é a diferença entre 'Quando ele chegar' e 'Quando ele chegará'? Dê exemplos.",
    ],
    "Gramática – Crase e Regência": [
        "Use crase ou não: 'Vou ___ Lisboa amanhã.' Justifique.",
        "Explique a diferença de regência: 'assistir o filme' vs 'assistir ao filme'.",
        "Corrija: 'Prefiro ir à pé do que de carro.' Justifique.",
    ],
    "Vocabulário e Idiomas": [
        "O que significa a expressão brasileira 'fazer vista grossa'? Use em uma frase.",
        "Qual a diferença de uso entre 'mau' e 'mal'? Dê três exemplos de cada.",
        "Explique a origem e uso da expressão 'dar o troco'.",
    ],
    "Conhecimento Cultural / Histórico PT": [
        "Quem foi Tiradentes e qual foi seu papel na história do Brasil?",
        "Explique o movimento literário do Modernismo brasileiro e cite dois autores centrais.",
        "O que foi a Revolução dos Cravos em Portugal?",
    ],
    "Raciocínio em Português": [
        "Se Pedro tem o dobro da idade de Maria, e a soma das idades deles é 36, quantos anos cada um tem? Explique passo a passo em português.",
        "Um trem parte às 8h a 90 km/h. Outro parte às 9h a 120 km/h na mesma direção. Em que hora o segundo alcança o primeiro? Resolva em português.",
    ],
    "Registro Formal / Redação": [
        "Escreva uma introdução de dissertação argumentativa sobre o impacto das redes sociais na democracia.",
        "Reescreva de forma formal: 'A gente precisa resolver esse pepino logo senão vai dar ruim.'",
    ],
    "Ortografia pós-Acordo": [
        "Após o Acordo Ortográfico de 1990, qual dessas grafias está correta: 'acção' ou 'ação'? Explique a regra.",
        "Escreva corretamente com o novo acordo: 'óptimo, facto, sector, directório'.",
    ],
}

# ---------------------------------------------------------------------------
# Perplexity reference corpus — diverse, naturally complex PT sentences
# ---------------------------------------------------------------------------
PERPLEXITY_CORPUS = [
    "A filosofia da linguagem investiga a relação entre as palavras, os pensamentos e o mundo exterior.",
    "O subjuntivo futuro é amplamente utilizado em orações condicionais e temporais na língua portuguesa.",
    "Apesar das dificuldades econômicas, o povo brasileiro manteve sua capacidade criativa e cultural.",
    "A Constituição Federal de 1988 estabeleceu os direitos e deveres fundamentais dos cidadãos brasileiros.",
    "O fenômeno da crase ocorre quando a preposição 'a' se funde com o artigo definido feminino 'a'.",
    "Os lusíadas, de Luís de Camões, é considerado o maior poema épico da língua portuguesa.",
    "A biodiversidade da Amazônia é reconhecida internacionalmente como patrimônio natural da humanidade.",
    "O samba, originário dos terreiros afro-brasileiros, tornou-se símbolo da identidade cultural do Brasil.",
    "A transição demográfica brasileira reflete mudanças profundas nos padrões de natalidade e mortalidade.",
    "A concordância verbal em português exige que o verbo concorde em número e pessoa com o sujeito.",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


MAX_GPU_MEM = "5GiB"   # cap GPU usage; rest spills to CPU RAM


def load_base():
    print(f"  Loading base model ({BASE_MODEL}) …")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config(),
        device_map="auto",
        max_memory={0: MAX_GPU_MEM, "cpu": "32GiB"},
        low_cpu_mem_usage=True,
    )
    mdl.eval()
    return mdl, tok


def load_adapter():
    print(f"  Loading base + adapter from {ADAPTER_PATH} …")
    tok = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config(),
        device_map="auto",
        max_memory={0: MAX_GPU_MEM, "cpu": "32GiB"},
        low_cpu_mem_usage=True,
    )
    mdl = PeftModel.from_pretrained(base, ADAPTER_PATH)
    mdl.eval()
    return mdl, tok


def unload(model):
    """Free GPU memory before loading the next model."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------
@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str]) -> float:
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        out = model(ids, labels=ids)
        n   = ids.shape[-1] - 1          # exclude first token from count
        total_nll    += out.loss.item() * n
        total_tokens += n
    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
GEN_KWARGS = dict(
    max_new_tokens=300,
    temperature=0.3,      # low temp for more deterministic, comparable outputs
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)

SYSTEM_PT = "Você é um assistente especializado em língua portuguesa. Responda sempre em português, de forma precisa e clara."


def chat_generate(model, tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PT},
        {"role": "user",   "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids      = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **GEN_KWARGS,
        )

    new_tokens = out[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
W = 80   # terminal wrap width

def separator(char="─", n=W):
    print(char * n)

def section(title: str):
    print()
    separator("═")
    print(f"  {title}")
    separator("═")

def wrap(text: str, indent=4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=W - indent, initial_indent=prefix, subsequent_indent=prefix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_all_prompts(model, tokenizer) -> dict[str, list[str]]:
    """Generate responses for every eval prompt. Returns {category: [answer, ...]}."""
    answers = {}
    for category, prompts in EVAL_PROMPTS.items():
        answers[category] = []
        for prompt in prompts:
            answers[category].append(chat_generate(model, tokenizer, prompt))
    return answers


RESULTS_BASE    = "/tmp/eval_base.json"
RESULTS_ADAPTER = "/tmp/eval_adapter.json"


def run_worker(mode: str, out_path: str):
    """Load one model, run all evals, dump results to JSON, exit."""
    if mode == "base":
        model, tok = load_base()
    else:
        model, tok = load_adapter()

    ppl = perplexity(model, tok, PERPLEXITY_CORPUS)
    answers = run_all_prompts(model, tok)

    with open(out_path, "w") as f:
        json.dump({"perplexity": ppl, "answers": answers}, f)

    print(f"\n  Results saved to {out_path}")


def spawn_worker(mode: str, out_path: str):
    """Run a fresh Python process for one model to fully reset CUDA memory."""
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.run(
        [sys.executable, __file__, f"--worker={mode}", f"--out={out_path}"],
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Worker '{mode}' failed (exit {proc.returncode})")


def main():
    print("\n" + "═" * W)
    print("  BENCHMARK: Qwen2.5-3B  vs  Qwen2.5-3B + Portuguese Adapter")
    print("═" * W + "\n")

    # ── Phase 1: BASE model (isolated subprocess) ─────────────────────────────
    section("PHASE 1 / 2 — BASE MODEL (Qwen2.5-3B, no adapter)")
    spawn_worker("base", RESULTS_BASE)

    # ── Phase 2: ADAPTER model (fresh subprocess, full GPU reset) ─────────────
    section("PHASE 2 / 2 — ADAPTER MODEL (Qwen2.5-3B + PT adapter)")
    spawn_worker("adapter", RESULTS_ADAPTER)

    # ── Load results ──────────────────────────────────────────────────────────
    with open(RESULTS_BASE) as f:
        base_data = json.load(f)
    with open(RESULTS_ADAPTER) as f:
        ada_data = json.load(f)

    ppl_base    = base_data["perplexity"]
    ppl_ada     = ada_data["perplexity"]
    base_answers = base_data["answers"]
    ada_answers  = ada_data["answers"]

    # ── Side-by-side display ──────────────────────────────────────────────────
    section("QUALITATIVE: side-by-side prompt evaluation")

    for category, prompts in EVAL_PROMPTS.items():
        print(f"\n{'─'*W}")
        print(f"  CATEGORY: {category}")
        print(f"{'─'*W}")

        for i, (prompt, base_ans, ada_ans) in enumerate(
            zip(prompts, base_answers[category], ada_answers[category]), 1
        ):
            print(f"\n  [{i}] {prompt}")
            separator("·")
            print("\n  BASE MODEL:")
            print(wrap(base_ans))
            print("\n  ADAPTER MODEL:")
            print(wrap(ada_ans))

    # ── Final summary ─────────────────────────────────────────────────────────
    delta    = ppl_base - ppl_ada
    improved = delta > 0
    pct      = abs(delta) / ppl_base * 100

    section("FINAL SUMMARY")
    print()
    print(f"  Perplexity on Portuguese reference corpus (↓ better)")
    print(f"    Base model : {ppl_base:.4f}")
    print(f"    Adapter    : {ppl_ada:.4f}")
    print(f"    Delta      : {delta:+.4f}  ({pct:.1f}% {'improvement ✓' if improved else 'regression ✗'})")
    print()
    if improved:
        print("  → Fine-tuning improved Portuguese language modelling.")
    else:
        print("  → Fine-tuning did not improve perplexity (check training duration / data quality).")
    print()
    print("  Review side-by-side outputs above for qualitative assessment.")
    print("  Focus on: subjunctive, crase, idioms, cultural facts, formal register.")
    separator("═")
    print()


if __name__ == "__main__":
    # Worker mode: called by spawn_worker() as a subprocess
    worker_mode  = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--worker=")), None)
    worker_out   = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    display_only = "--display-only" in sys.argv

    if worker_mode:
        run_worker(worker_mode, worker_out)
    elif display_only:
        # Both JSON files already exist — just render the report
        with open(RESULTS_BASE) as f:
            base_data = json.load(f)
        with open(RESULTS_ADAPTER) as f:
            ada_data = json.load(f)

        ppl_base     = base_data["perplexity"]
        ppl_ada      = ada_data["perplexity"]
        base_answers = base_data["answers"]
        ada_answers  = ada_data["answers"]

        print("\n" + "═" * W)
        print("  BENCHMARK: Qwen2.5-3B  vs  Qwen2.5-3B + Portuguese Adapter")
        print("═" * W + "\n")

        section("QUALITATIVE: side-by-side prompt evaluation")
        for category, prompts in EVAL_PROMPTS.items():
            print(f"\n{'─'*W}")
            print(f"  CATEGORY: {category}")
            print(f"{'─'*W}")
            for i, (prompt, base_ans, ada_ans) in enumerate(
                zip(prompts, base_answers[category], ada_answers[category]), 1
            ):
                print(f"\n  [{i}] {prompt}")
                separator("·")
                print("\n  BASE MODEL:")
                print(wrap(base_ans))
                print("\n  ADAPTER MODEL:")
                print(wrap(ada_ans))

        delta    = ppl_base - ppl_ada
        improved = delta > 0
        pct      = abs(delta) / ppl_base * 100

        section("FINAL SUMMARY")
        print()
        print(f"  Perplexity on Portuguese reference corpus (↓ better)")
        print(f"    Base model : {ppl_base:.4f}")
        print(f"    Adapter    : {ppl_ada:.4f}")
        print(f"    Delta      : {delta:+.4f}  ({pct:.1f}% {'improvement ✓' if improved else 'regression ✗'})")
        print()
        if improved:
            print("  → Fine-tuning improved Portuguese language modelling.")
        else:
            print("  → Fine-tuning did not improve perplexity (check training duration / data quality).")
        print()
        print("  Review side-by-side outputs above for qualitative assessment.")
        print("  Focus on: subjunctive, crase, idioms, cultural facts, formal register.")
        separator("═")
        print()
    else:
        main()
