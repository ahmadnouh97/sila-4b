# SILA-4B

SILA-4B tests whether a small Arabic-focused QLoRA adapter improves tool calling while preserving the base model's English performance.

## Current status

The data, leakage audit, local environment, and notebook are prepared. The owner stopped the training attempt before its first epoch checkpoint and plans to run the notebook manually. **There is no trained adapter or performance result yet.** Benchmark loaders and scoring functions exist; an end-to-end model evaluation runner remains to be built.

## Experiment and success criteria

Compare the untouched base model with its Arabic-only adapter using the same prompts, tools, decoding settings, and quantization. Training uses 5,280 synthetic Arabic examples; 1,320 held-out examples are used for training diagnostics. The evaluation sets are never used for training.

The experiment succeeds only if the adapter meets all three criteria:

1. Arabic exact tool-call accuracy improves by **at least 5 percentage points**, or relative error falls by **at least 25%**.
2. English BFCL accuracy falls by **no more than 2 percentage points**.
3. Arabic no-call accuracy does not decrease.

Relative error reduction is `(base error - adapter error) / base error`.

The primary Arabic evaluation is the owner-reviewed 100-case set. ArabFuncBench (1,000 cases) provides an external Arabic comparison. BFCL V4 uses a fixed 500-case English subset: 400 `simple_python` and 100 `multiple`. Score BFCL with its [official AST evaluator](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py) and original records.

No performance results are available yet. The local leakage audit checks exact and normalized prompts, tool names, full schemas, and expected calls; it cannot rule out semantic similarity. The audit report is `reports/leakage_audit.json` and is ignored by Git along with local data and outputs.

## Code guide

| File | What it does |
| --- | --- |
| `src/sila/generate_training_corpus.py` | Creates Arabic training examples, separates training and check sets, and records file hashes. Regenerating replaces those files. |
| `src/sila/validate_stress_set.py` | Checks required fields, categories, tool input descriptions, and review status in the Arabic evaluation file. It cannot judge translation quality. |
| `src/sila/audit_leakage.py` | Checks whether training examples repeat evaluation prompts, tools, or answers; writes the audit report. |
| `src/sila/benchmarks.py` | Reads local ArabFuncBench and BFCL files into the project's common format. It does not download data or score model outputs. |
| `src/sila/evaluation.py` | Compares model responses with expected tool calls and calculates accuracy metrics. It never runs a tool call. |
| `src/sila/schemas.py` | Defines and validates the common data formats used by the other modules. |
| `notebooks/train_qwen_qlora.ipynb` | Loads the pinned model in 4-bit mode, checks the training gates, and runs QLoRA fine-tuning. It saves the adapter and run record under `outputs/`. |

## Run locally

Use Python 3.14, `uv`, and the repository `.venv`. The training kernel is named `Python (sila-4b .venv)`.

```powershell
uv sync --group training
$env:PYTHONPATH = "src"
uv run python -m sila.validate_stress_set data/stress_test.jsonl --final
uv run python -m sila.audit_leakage
```

Open the notebook with the `Python (sila-4b .venv)` kernel and run its cells from top to bottom. It reports the expected step count and live training progress. The full run takes hours on the local 8 GB RTX 4060. Do not rerun the corpus generator unless you intend to replace the current generated data.

Development checks:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Data and reproducibility

- Base model: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Training uses bitsandbytes NF4 4-bit quantization with LoRA.
- ArabFuncBench: revision `c11e4e5ede503892b85633c40c16720c75310e48`; licensed CC BY 4.0. Keep it evaluation-only and use the [upstream metrics](https://github.com/lsadouk/ArabFuncBench) for its benchmark scores.
- BFCL V4: Gorilla revision `9d8416a96d1d69975493f1b6d60ff07d12a1726a`; Apache 2.0. Only the fixed subset described above is in scope.
- Training corpus: first-party generator v1.7. Its manifest pins the local data hashes and split details.
- JAIS (`inception42/jais-13b-chat`) is a planned comparator. Its immutable revision, license, and custom code still need review before download or use.

Local datasets, model outputs, and reports are ignored by Git. Preserve raw model responses for scoring, and never execute generated tool calls.
