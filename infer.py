"""
Interactive inference script for the Portuguese QLoRA fine-tuned model.
Base: Qwen/Qwen2.5-3B  |  Adapter: models/qwen-portuguese-v0001
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER_PATH = "./models/qwen-portuguese-v0001"
BASE_MODEL   = "Qwen/Qwen2.5-3B"

# Generation defaults — tweak freely
DEFAULTS = dict(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)


def load_model():
    print(f"Loading base model  : {BASE_MODEL}")
    print(f"Loading LoRA adapter: {ADAPTER_PATH}\n")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, prompt: str, system: str = "Você é um assistente útil.", **kwargs) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    params = {**DEFAULTS, **kwargs}

    with torch.no_grad():
        output_ids = model.generate(input_ids, **params)

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    model, tokenizer = load_model()

    print("=" * 60)
    print("Modelo pronto. Digite sua mensagem em português.")
    print("Comandos: 'sair' para encerrar, 'sistema:<texto>' para mudar o system prompt.")
    print("=" * 60 + "\n")

    system = "Você é um assistente útil que responde sempre em português."

    while True:
        try:
            user_input = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("sair", "exit", "quit"):
            print("Até logo!")
            break

        if user_input.lower().startswith("sistema:"):
            system = user_input[len("sistema:"):].strip()
            print(f"[System prompt atualizado: {system}]\n")
            continue

        response = generate(model, tokenizer, user_input, system=system)
        print(f"\nModelo: {response}\n")


if __name__ == "__main__":
    main()
