# Language Training Data

Each sub-folder corresponds to one language/locale using the
`<ISO-639-1>-<ISO-3166-1>` format (e.g. `pt-BR`, `en-US`, `fr-FR`).

## Adding a new language

1. Create a folder: `training_data/languages/<lang-code>/`
2. Drop your corpus files in any supported format (`.txt`, `.csv`, `.json`, `.jsonl`, `.pdf`)
3. Load it:
   ```python
   from data.ingestion import load_folder
   ds = load_folder("training_data/languages/pt-BR/")
   ```
4. Pass the dataset to `factory.py`:
   ```bash
   python factory.py train \
     --model Qwen/Qwen2.5-7B \
     --dataset-dir training_data/languages/pt-BR/ \
     --metric perplexity \
     --budget 3600
   ```

## Recommended data sources by language

| Language | Source | HuggingFace ID |
|----------|--------|---------------|
| Brazilian Portuguese | Wikipedia PT | `wikimedia/wikipedia` `20231101.pt` |
| Brazilian Portuguese | mC4 | `mc4` `pt` |
| French | Wikipedia FR | `wikimedia/wikipedia` `20231101.fr` |
| Spanish | Wikipedia ES | `wikimedia/wikipedia` `20231101.es` |
| English | OpenWebText | `Skylion007/openwebtext` |
| Chinese | Wikipedia ZH | `wikimedia/wikipedia` `20231101.zh` |

The `data/portuguese.py` module already has loaders for Brazilian Portuguese.
Implement similar modules in `data/` for other languages and register them
with `@register_dataset(name)`.
