# SLM Training Factory

> *One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. —@karpathy, March 2026*

A **modular factory for fine-tuning small language models** on consumer hardware (6 GB VRAM). It uses QLoRA to adapt any HuggingFace model to a new language or domain — and an autonomous AI research loop to find the best hyperparameters overnight, no PhD required.

---

## How it works

The factory is built around a three-phase strategy:

```
Phase 1 — Hyperparameter Search   (hours)
Phase 2 — Full Language Training  (hours → overnight)
Phase 3 — Domain / Task Fine-tune (optional, hours)
```

Each phase runs the same training loop, but with a different goal. The key innovation is **Phase 1: letting an AI agent do the hyperparameter search for you**.

---

## Why the micro-experiment loop is a great idea

Hyperparameter tuning is the worst part of fine-tuning. The search space is large (learning rate, LoRA rank, sequence length, batch size, scheduler…), experiments are slow, and intuition from large-scale training often doesn't transfer to QLoRA on 6 GB VRAM.

The approach here:

1. **Run short cheap experiments** — each one is capped at a time budget (e.g. 30 minutes). This is just enough to get a meaningful perplexity signal.
2. **Let the AI agent propose changes** — after each experiment, the agent reads the results and proposes one config mutation (increase `r`, halve the learning rate, try a longer sequence, etc.).
3. **Keep only improvements** — if the metric didn't improve, the config change is reverted. If it did, the adapter is versioned and the new config becomes the baseline.
4. **Run overnight** — in 8–10 hours you get 16–20 experiments, and typically converge on a config that would have taken days to find manually.

This is the same philosophy as neural architecture search and AutoML, applied to fine-tuning. The agent doesn't need to understand the math — it reads the metric trend and applies heuristics. The human just wakes up and reads the results.

**Cost:** a 1-hour autoresearch session (4–5 experiments) costs ~$0.10 in Claude API calls. The GPU is the bottleneck, not the API.

---

## The three phases

### Phase 1 — Hyperparameter search

Run short experiments on a small data sample to find the best config for your hardware and target language.

```bash
# 1-hour autoresearch: tries 4–5 configs, keeps the best
python autoresearch/loop.py --config config/portuguese.yaml
```

What the agent tunes:
- LoRA rank `r` and `alpha`
- Learning rate and scheduler
- Sequence length (`max_seq_len`)
- Batch size and gradient accumulation

What you get: a `config/portuguese_phase2.yaml` (or similar) with the winning hyperparameters, ready for Phase 2.

### Phase 2 — Full language training

Run the winning config on the full dataset with a larger time budget. This is where the model actually learns to speak the language.

```bash
# 4-hour full training run with best hyperparameters
python factory.py --config config/portuguese_phase2.yaml

# After it finishes, merge the LoRA adapter into the base model
python factory.py merge --config config/portuguese_phase2.yaml
```

Output: a standalone merged model (e.g. `./models/qwen-portuguese-v1`) ready for inference or Phase 3.

### Phase 3 — Domain / task fine-tuning (optional)

Use the Phase 2 model as the base for specialization: legal reasoning, medical summarization, code generation in your language, etc.

```bash
# Point a new config at the merged Phase 2 model
cp config/portuguese_phase2.yaml config/my-domain.yaml
# Edit: model.name: ./models/qwen-portuguese-v1
#       dataset.path: training_data/my-domain/
#       training.budget_seconds: 7200

python autoresearch/loop.py --config config/my-domain.yaml
python factory.py --config config/my-domain.yaml  # full run after search
python factory.py merge --config config/my-domain.yaml
```

---

## Installation

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), NVIDIA GPU ≥6 GB VRAM

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/your-org/slm-factory
cd slm-factory
uv sync
```

For the AI research loop (Phase 1), you need an Anthropic API key:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

---

## Quick start

```bash
# Verify setup (no GPU required)
python factory.py --config config/portuguese.yaml --dry-run

# Phase 1: find best hyperparameters (1 hour, ~4 experiments)
python autoresearch/loop.py --config config/portuguese.yaml

# Phase 2: train with the best config on the full dataset (4 hours)
python factory.py --config config/portuguese_phase2.yaml

# Merge the adapter into a standalone model
python factory.py merge --config config/portuguese_phase2.yaml

# Use the model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./models/qwen-portuguese-v1')
tok   = AutoTokenizer.from_pretrained('./models/qwen-portuguese-v1')
out   = model.generate(**tok('Olá, como vai', return_tensors='pt'), max_new_tokens=50)
print(tok.decode(out[0], skip_special_tokens=True))
"
```

---

## Adding a new language or domain

1. **Create a config** by copying the closest example:

```bash
cp config/portuguese.yaml config/french.yaml
```

2. **Edit the config:**

```yaml
model:
  name: Qwen/Qwen2.5-3B      # any HuggingFace causal LM

lora:
  r: 16                       # start low; Phase 1 will find the best value
  alpha: 32

training:
  budget_seconds: 1800        # 30 min per experiment for Phase 1 search
  max_seq_len: 512
  learning_rate: 2.0e-4
  output_dir: ./checkpoints/qwen-french

dataset:
  name: wikipedia-pt          # replace with your dataset (see below)

metric:
  name: perplexity
  lower_is_better: true
```

3. **Pick a dataset.** Built-in HuggingFace loaders:

| Name | Description |
|------|-------------|
| `wikipedia-pt` | Wikipedia in Brazilian Portuguese (~1.1M articles) |
| `mc4-pt` | mC4 Portuguese web text |
| `portuguese-mix` | Wikipedia PT + mC4 PT combined |

To use any HuggingFace dataset, add it to `data/registry.py`:

```python
@register_dataset("wikipedia-fr")
def load_wikipedia_fr(max_samples=None, **kwargs):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train")
    if max_samples:
        ds = ds.select(range(max_samples))
    return ds  # must have a 'text' column
```

To use local files (`.txt`, `.jsonl`, `.csv`), set `dataset.path` in your config:

```yaml
dataset:
  path: training_data/languages/fr-FR/
```

4. **Run Phase 1** and let the agent find the best config:

```bash
python autoresearch/loop.py --config config/french.yaml
```

---

## VRAM guide

This factory targets **6 GB VRAM** (e.g. RTX 3050 6GB, RTX 3060). The key settings:

| Model | 4-bit VRAM | Fits in 6 GB? |
|-------|-----------|---------------|
| Qwen2.5-1.5B | ~1.5 GB | ✓ comfortable |
| Qwen2.5-3B | ~2.5 GB | ✓ good choice |
| Qwen2.5-7B | ~5.5 GB | ✗ OOM on RTX 3050 |

If you run out of memory during training, reduce these settings:

| Setting | Default | Reduce to |
|---------|---------|-----------|
| `max_seq_len` | 512 | 256 |
| `batch_size` | 1 | 1 (already minimum) |
| `lora.r` | 64 | 32 or 16 |
| `gradient_accumulation_steps` | 16 | 8 (reduces effective batch size) |

Always keep `load_in_4bit: true` and `gradient_checkpointing: true`.

For GPUs with 8+ GB VRAM you can use Qwen2.5-7B and increase `max_seq_len` to 1024.

---

## CLI reference

```bash
# Single training run
python factory.py --config CONFIG [--dry-run]

# Merge LoRA adapter into base model (creates standalone model)
python factory.py merge --config CONFIG
python factory.py merge --base BASE_MODEL --adapter ADAPTER_DIR --output OUTPUT

# View version history
python factory.py versions --config CONFIG

# List registered datasets
python factory.py list-datasets

# Autonomous hyperparameter search loop (Phase 1)
python autoresearch/loop.py --config CONFIG
```

---

## Project structure

```
factory.py               ← CLI entrypoint (train, merge, versions, list-datasets)
data_parser.py           ← Claude-powered: raw text → instruction-tuning pairs
config/
  base.yaml              ← shared defaults
  portuguese.yaml        ← Phase 1: hyperparameter search config
  portuguese_phase2.yaml ← Phase 2: full training with best hyperparameters
models/
  registry.py            ← HuggingFace model loader + 4-bit quantization
  adapters.py            ← LoRA adapter construction (PEFT)
data/
  registry.py            ← dataset name → loader mapping
  ingestion.py           ← generic file loader (TXT, CSV, JSON, JSONL, PDF)
  portuguese.py          ← Wikipedia PT, mC4 PT, OSCAR PT
training_data/           ← gitignored; drop your corpus files here
  languages/
    pt-BR/               ← Brazilian Portuguese corpus
    en-US/               ← English corpus
    <lang-code>/         ← add any language
trainers/
  base.py                ← abstract trainer interface
  qlora.py               ← QLoRA trainer (6 GB VRAM target)
metrics/
  perplexity.py          ← language model perplexity
  bpb.py                 ← bits-per-byte (vocab-independent)
versioning/
  model_store.py         ← metric-gated version store
autoresearch/
  loop.py                ← autonomous research loop
  agent_prompt.md        ← agent instructions for hyperparameter search
tests/                   ← unit + integration tests (no GPU required)
```

---

## Model versioning

Every time training improves the metric, a new adapter version is saved. The base model is never modified — only the LoRA weights (~200 MB per version).

```
checkpoints/qwen-portuguese/
├── v0001/              perplexity = 9.82
├── v0002/              perplexity = 8.41
└── v0003/              perplexity = 7.35  ← best
```

```bash
python factory.py versions --config config/portuguese.yaml
```

---

## Tests

```bash
# Unit tests — no GPU required, ~10 seconds
python -m pytest tests/ --ignore=tests/test_e2e.py \
                        --ignore=tests/test_autoresearch_wikipedia.py -v

# End-to-end: full training loop with tiny-gpt2 (needs internet + ~2 min)
python -m pytest tests/test_e2e.py -v -s

# Full validation: Wikipedia PT → LoRA → perplexity check (~5 min, GPU recommended)
python -m pytest tests/test_autoresearch_wikipedia.py -v -s
```

---

## License

MIT
