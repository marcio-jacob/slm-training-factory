"""
legal_eval.py — Deep evaluation of the judicial Portuguese adapter.

Three-way comparison:
  BASE       : Qwen/Qwen2.5-3B (no fine-tuning)
  PORTUGUESE : qwen-portuguese-v1 (Wikipedia PT adapter merged)
  JUDICIAL   : qwen-portuguese-v1 + judicial-v0001 LoRA adapter

Tests cover:
  1. CF/88 constitutional provisions
  2. Civil law (CC) — contracts, persons, property
  3. Criminal law (CP/CPP) — crimes, procedure
  4. Labor law (CLT) — employment rights
  5. Consumer law (CDC) — consumer protection
  6. Scenario-based legal reasoning
  7. Cross-code interpretation
  8. Procedural / deadlines
  9. Legal definitions

Each prompt is a natural legal text starter — the model continues it.
Uses subprocess isolation to avoid CUDA OOM between models.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python legal_eval.py
    python legal_eval.py --display-only
    python legal_eval.py --worker=base     --out=/tmp/legal_base.json
    python legal_eval.py --worker=portuguese --out=/tmp/legal_pt.json
    python legal_eval.py --worker=judicial --out=/tmp/legal_judicial.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "id": "cf_art5",
        "category": "CF/88 — Direitos Fundamentais",
        "prompt": (
            "A Constituição Federal de 1988, em seu artigo 5º, estabelece que todos são iguais "
            "perante a lei, sem distinção de qualquer natureza, garantindo-se aos brasileiros e "
            "aos estrangeiros residentes no País a inviolabilidade do direito à vida, à liberdade, "
            "à igualdade, à segurança e à propriedade, nos seguintes termos:"
        ),
    },
    {
        "id": "cf_art37",
        "category": "CF/88 — Administração Pública",
        "prompt": (
            "A administração pública direta e indireta de qualquer dos Poderes da União, dos "
            "Estados, do Distrito Federal e dos Municípios obedecerá aos princípios de legalidade, "
            "impessoalidade, moralidade, publicidade e eficiência, e também ao seguinte:"
        ),
    },
    {
        "id": "cc_contratos",
        "category": "Código Civil — Contratos",
        "prompt": (
            "O contrato é válido quando celebrado por agente capaz, versar sobre objeto lícito, "
            "possível, determinado ou determinável, e revestir a forma prescrita ou não defesa em "
            "lei. A nulidade do contrato ocorre quando"
        ),
    },
    {
        "id": "cc_responsabilidade",
        "category": "Código Civil — Responsabilidade Civil",
        "prompt": (
            "Aquele que, por ação ou omissão voluntária, negligência ou imprudência, violar direito "
            "e causar dano a outrem, ainda que exclusivamente moral, comete ato ilícito. A "
            "obrigação de reparar o dano independe de culpa nos casos"
        ),
    },
    {
        "id": "cp_homicidio",
        "category": "Código Penal — Homicídio",
        "prompt": (
            "O homicídio simples, previsto no artigo 121 do Código Penal, consiste em matar alguém, "
            "sendo cominada pena de reclusão de 6 a 20 anos. O homicídio é qualificado, com pena "
            "de reclusão de 12 a 30 anos, quando"
        ),
    },
    {
        "id": "cp_legitima_defesa",
        "category": "Código Penal — Excludentes de Ilicitude",
        "prompt": (
            "Entende-se em legítima defesa quem, usando moderadamente dos meios necessários, "
            "repele injusta agressão, atual ou iminente, a direito seu ou de outrem. O excesso "
            "punível ocorre quando o agente"
        ),
    },
    {
        "id": "clt_rescisao",
        "category": "CLT — Rescisão do Contrato de Trabalho",
        "prompt": (
            "A rescisão do contrato de trabalho sem justa causa pelo empregador obriga ao pagamento "
            "das seguintes verbas rescisórias ao empregado: saldo de salário, aviso prévio, "
            "férias proporcionais acrescidas de um terço, décimo terceiro proporcional,"
        ),
    },
    {
        "id": "clt_ferias",
        "category": "CLT — Férias",
        "prompt": (
            "Todo empregado terá direito anualmente ao gozo de um período de férias, sem prejuízo "
            "da remuneração, na seguinte proporção: 30 dias corridos quando não houver faltado ao "
            "serviço mais de 5 vezes; 24 dias corridos quando houver tido de 6 a 14 faltas;"
        ),
    },
    {
        "id": "cdc_defeito",
        "category": "CDC — Responsabilidade por Defeito",
        "prompt": (
            "O fornecedor de serviços responde independentemente da existência de culpa pela "
            "reparação dos danos causados aos consumidores por defeitos relativos à prestação dos "
            "serviços. O fornecedor só não será responsabilizado quando provar que"
        ),
    },
    {
        "id": "cdc_prazo_garantia",
        "category": "CDC — Prazo de Reclamação",
        "prompt": (
            "O direito de reclamar pelos vícios aparentes ou de fácil constatação caduca em: "
            "trinta dias, tratando-se de fornecimento de serviço e de produtos não duráveis; "
            "noventa dias, tratando-se de fornecimento de serviço e de produtos duráveis. "
            "O prazo decadencial começa a correr a partir"
        ),
    },
    {
        "id": "scenario_demissao",
        "category": "Cenário — Direito do Trabalho",
        "prompt": (
            "João trabalha há 8 anos na mesma empresa, com carteira assinada, recebendo salário "
            "de R$ 4.000,00 mensais. Foi demitido sem justa causa. Nos termos da Consolidação das "
            "Leis do Trabalho e da legislação complementar, João tem direito a receber"
        ),
    },
    {
        "id": "scenario_consumidor",
        "category": "Cenário — Direito do Consumidor",
        "prompt": (
            "Maria comprou uma televisão há 6 meses. O aparelho apresentou defeito de fabricação "
            "que a impossibilita de funcionar. O fornecedor se recusou a trocar o produto. "
            "Com base no Código de Defesa do Consumidor, Maria pode exigir, à sua escolha,"
        ),
    },
    {
        "id": "scenario_penal",
        "category": "Cenário — Direito Penal",
        "prompt": (
            "Carlos, primário e de bons antecedentes, foi condenado por furto simples de um "
            "celular avaliado em R$ 800,00. O juiz, ao aplicar a pena, deve considerar o princípio "
            "da insignificância, pois o Supremo Tribunal Federal consolidou entendimento de que"
        ),
    },
    {
        "id": "lgpd_dados",
        "category": "LGPD — Proteção de Dados",
        "prompt": (
            "A Lei Geral de Proteção de Dados Pessoais estabelece que o tratamento de dados "
            "pessoais somente poderá ser realizado quando o titular consentir. O consentimento "
            "será nulo quando as informações fornecidas ao titular tiverem conteúdo enganoso ou "
            "abusivo, ou quando"
        ),
    },
    {
        "id": "eca_crianca",
        "category": "ECA — Proteção à Criança",
        "prompt": (
            "É dever da família, da comunidade, da sociedade em geral e do poder público assegurar, "
            "com absoluta prioridade, a efetivação dos direitos referentes à vida, à saúde, à "
            "alimentação, à educação, ao esporte, ao lazer, à profissionalização, à cultura, à "
            "dignidade, ao respeito, à liberdade e à convivência familiar. A criança e o "
            "adolescente gozam de todos os direitos fundamentais inerentes à pessoa humana, "
            "sem prejuízo da proteção integral"
        ),
    },
]

# Generation parameters — slightly lower temperature for legal text (precision matters)
GEN_KWARGS = dict(
    max_new_tokens=200,
    temperature=0.6,
    top_p=0.90,
    repetition_penalty=1.1,
    do_sample=True,
)

BASE_MODEL    = "Qwen/Qwen2.5-3B"
PT_MODEL      = "./models/qwen-portuguese-v1"
JUDICIAL_ADAPTER = "./models/qwen-judicial-versions/v0001"

RESULTS_BASE      = "/tmp/legal_base.json"
RESULTS_PORTUGUESE = "/tmp/legal_pt.json"
RESULTS_JUDICIAL  = "/tmp/legal_judicial.json"


# ---------------------------------------------------------------------------
# Worker — runs in isolated subprocess
# ---------------------------------------------------------------------------

def worker_main(model_type: str, out_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    if model_type == "base":
        model_path = BASE_MODEL
        print(f"  Loading base model {BASE_MODEL} …")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="cpu", trust_remote_code=True,
        )
        model = model.to("cuda:0")

    elif model_type == "portuguese":
        from pathlib import Path
        model_path = str(Path(PT_MODEL).resolve())
        print(f"  Loading Portuguese model {model_path} …")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="cpu", trust_remote_code=True,
        )
        model = model.to("cuda:0")

    elif model_type == "judicial":
        from pathlib import Path
        from peft import PeftModel
        model_path = str(Path(PT_MODEL).resolve())
        adapter_path = str(Path(JUDICIAL_ADAPTER).resolve())
        print(f"  Loading Portuguese base {model_path} …")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="cpu", trust_remote_code=True,
        )
        base = base.to("cuda:0")
        print(f"  Attaching judicial adapter {adapter_path} …")
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    results = []

    for item in PROMPTS:
        print(f"  [{model_type}] {item['id']} …")
        inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **GEN_KWARGS,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append({"id": item["id"], "continuation": continuation.strip()})
        print(f"    done ({len(new_tokens)} tokens)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Orchestrator — launches one subprocess per model
# ---------------------------------------------------------------------------

def run_worker(model_type: str, out_path: str):
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [sys.executable, __file__, f"--worker={model_type}", f"--out={out_path}"]
    print(f"\n{'='*70}")
    print(f"  Running worker: {model_type.upper()}")
    print(f"{'='*70}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"  [ERROR] Worker {model_type} exited with code {proc.returncode}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 76, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def display_results():
    for path, label in [
        (RESULTS_BASE, "BASE (Qwen2.5-3B, no fine-tuning)"),
        (RESULTS_PORTUGUESE, "PORTUGUESE (Wikipedia PT adapter)"),
        (RESULTS_JUDICIAL, "JUDICIAL (judicial-v0001 adapter)"),
    ]:
        try:
            with open(path, encoding="utf-8") as f:
                data = {r["id"]: r["continuation"] for r in json.load(f)}
        except FileNotFoundError:
            data = {}
        globals()[label] = data

    base_data = {}
    pt_data   = {}
    jud_data  = {}
    for path, target in [
        (RESULTS_BASE, "base_data"),
        (RESULTS_PORTUGUESE, "pt_data"),
        (RESULTS_JUDICIAL, "jud_data"),
    ]:
        try:
            with open(path, encoding="utf-8") as f:
                d = {r["id"]: r["continuation"] for r in json.load(f)}
                if target == "base_data": base_data = d
                elif target == "pt_data": pt_data = d
                else: jud_data = d
        except FileNotFoundError:
            pass

    sep = "═" * 78

    print(f"\n{sep}")
    print("  LEGAL EVALUATION — 3-WAY COMPARISON")
    print(f"  Base · Portuguese · Judicial")
    print(sep)

    for item in PROMPTS:
        pid = item["id"]
        print(f"\n{'-'*78}")
        print(f"  CATEGORY: {item['category']}")
        print(f"{'-'*78}")
        print()
        print("  PROMPT (continuation start):")
        print(_wrap(item["prompt"]))
        print()

        for label, data in [
            ("BASE", base_data),
            ("PORTUGUESE", pt_data),
            ("JUDICIAL", jud_data),
        ]:
            text = data.get(pid, "(not run)")
            print(f"  [{label}]")
            print(_wrap(text))
            print()

    print(sep)
    print("  What to look for:")
    print("   • Does JUDICIAL correctly cite articles and provisions?")
    print("   • Does JUDICIAL use proper legal terminology (inciso, parágrafo único)?")
    print("   • Do the scenarios produce legally accurate continuations?")
    print("   • Does JUDICIAL outperform BASE and PORTUGUESE on legal precision?")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["base", "portuguese", "judicial"])
    parser.add_argument("--out", help="Output JSON path (worker mode)")
    parser.add_argument("--display-only", action="store_true")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker, args.out)
    elif args.display_only:
        display_results()
    else:
        run_worker("base",       RESULTS_BASE)
        run_worker("portuguese", RESULTS_PORTUGUESE)
        run_worker("judicial",   RESULTS_JUDICIAL)
        display_results()
