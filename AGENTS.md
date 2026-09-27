# Repository Guidelines

## Project Structure & Module Organization

- `src/sila/` contains the Python package. Keep reusable training, evaluation, and data-processing code here.
- `tests/` contains pytest tests; `tests/test_smoke.py` currently verifies that the package imports correctly.
- `data/`, `outputs/`, and `reports/` are tracked with `.gitkeep`, while their generated contents are ignored. Do not commit datasets, model outputs, or generated reports unless the project policy changes.
- `README.md` defines the experiment, metrics, leakage policy, and success criteria. Treat it as the source of truth for research decisions.

## Build, Test, and Development Commands

The project uses Python 3.14 and `uv`; dependencies are pinned in `uv.lock`.

```powershell
uv sync                 # Create/update the environment from the lockfile
uv run pytest           # Run all tests under tests/
uv run ruff check .     # Check E, F, and import-order rules
uv run ruff format .    # Format Python files
```

Run tests and lint checks before opening a pull request. This package has no separate build step yet.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Add type annotations to public functions and keep modules narrowly focused. Ruff targets Python 3.14 and enforces pycodestyle errors, Pyflakes, and sorted imports.

## Testing Guidelines

Use pytest. Name files `test_*.py` and tests `test_<behavior>`. Mirror package areas where practical, and add the smallest regression test that proves each non-trivial change. Tests must be deterministic and must not download models, execute generated tool calls, or depend on untracked local data.

## Commit & Pull Request Guidelines

History currently uses an imperative Conventional Commit prefix, for example `chore: initialize sila-4b experiment`. Continue with concise subjects such as `feat: add call parser` or `test: cover invalid responses`.

Pull requests should explain the change and its experimental impact, link any issue, and list validation commands run. Include relevant metric tables for experiment changes and screenshots only when presentation output changes. Keep generated artifacts and secrets out of commits.

## Reproducibility & Data Safety

Pin model and dataset revisions, preserve raw responses before parsing, and keep training data disjoint from evaluation prompts, schemas, and derived examples. Never commit credentials or private datasets; pass secrets through environment variables.

## SILA-4B Experiment Decisions and Gates

- The selected base is `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Use bitsandbytes 4-bit quantization with QLoRA, in a local Jupyter notebook. Verify the local GPU/runtime fit before a full training run; the recorded laptop GPU has 8 GB VRAM.
- The intended comparison remains frozen Qwen vs. its Arabic-only synthetic-data adapter, with `inception42/jais-13b-chat` as the previously agreed comparator on ArabFuncBench. Retain the README-required 100-case Arabic set and fixed 500-case BFCL V4 subset unless the owner changes scope.
- The owner says the current 100-case Arabic evaluation set has been reviewed by them. Its local rows may still carry draft statuses; before freezing or evaluating, record that review, set the final statuses, run `uv run python -m sila.validate_stress_set data/stress_test.jsonl --final`, and record the frozen hash. Do not edit the reviewed content without owner-provided corrections.
- Generator v1.7 and the live `data/training.manifest.json` plus `reports/training_corpus_review.md` are authoritative for current training artifacts. Older hashes in `reports/phase1.md` are stale. Keep all generated data and reports ignored and out of commits.
- Before any training run, obtain immutable local ArabFuncBench exports and BFCL V4 prompt/`possible_answer` files, record their revisions, and complete the leakage audit against the current train/validation corpus and frozen Arabic set. Keep benchmark data evaluation-only. Use BFCL's official AST evaluator for its score.
- JAIS is gated and requires an approved immutable revision plus review of its license and custom code before download or execution. Preserve raw model outputs and metadata; never execute generated tool calls.
- The project environment previously could not launch its Python 3.14 interpreter and had no ML training stack. Once implementation is authorized, establish and verify a compatible local Jupyter kernel; do not assume the current project environment is ready for model work.

## Implementation Authorization

Do not begin implementation, install the training stack, download models or benchmark data, or run training until the owner explicitly says **“start implementation.”** Repository review and this documentation update do not authorize those actions.
