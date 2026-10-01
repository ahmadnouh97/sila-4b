# Repository Guidelines

## Project

Read `README.md` for the experiment design, success criteria, current status, and code guide. Keep it concise and update it when those facts change.

- Python 3.14 project managed with `uv`; use the repository `.venv` and Jupyter kernel `Python (sila-4b .venv)`.
- `src/sila/` contains data generation, validation, benchmark loading, leakage audit, schemas, and scoring. `notebooks/train_qwen_qlora.ipynb` contains the local QLoRA workflow.
- Keep local data, model outputs, and reports out of Git. The reviewed evaluation set and benchmark exports are local and ignored.

## Experiment rules

- Base: `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`; use bitsandbytes NF4 4-bit quantization with LoRA.
- Training data is first-party Arabic synthetic data from generator v1.7. `data/training.manifest.json` is authoritative; do not regenerate it unless requested because generation replaces the current files.
- Evaluation data is evaluation-only. The 100 Arabic cases were reviewed by the owner; do not change their prompts, schemas, labels, or dates without owner corrections. The current leakage audit passed its exact/normalized checks; it does not detect all semantic similarity.
- Preserve raw model responses. Never execute generated tool calls. Use BFCL's official AST evaluator for BFCL scores.
- Training and paired evaluation are complete. The saved adapter failed the preset criteria; README.md records the aggregate results and limitations. Preserve the local artifacts. Do not launch training unless explicitly asked.
- JAIS remains gated. Do not download or use it until its immutable revision, license, and custom code are reviewed and approved.

## Development

```powershell
uv sync --group training
uv run pytest
uv run ruff check .
uv run ruff format .
```

Set `$env:PYTHONPATH = "src"` before running project modules with `uv run python -m sila.<module>`.
Use four-space indentation, type annotations on public functions, and `snake_case` for Python names. Keep tests deterministic and independent of network downloads and local ignored data.
