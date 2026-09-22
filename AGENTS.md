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
