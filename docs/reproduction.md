# Reproduction guide

[Project overview](../README.md)

The recorded experiment is complete and failed its acceptance criteria. Full reproduction requires local inputs and model artifacts that are not published here. These instructions document the workflow; training remains a manual, explicitly requested step.

## Recorded inputs and selection

The v1.7 corpus has 5,280 training rows and 1,320 diagnostic validation rows. The primary Arabic evaluation has 70 call and 30 no-call cases; the fixed BFCL subset has 400 `simple_python` and 100 `multiple` cases. ArabFuncBench supplies 1,000 external Arabic cases. Evaluation data is never used for training or checkpoint selection.

The three-epoch run saved an adapter identical in tensor values to `checkpoint-330`, the epoch-one checkpoint selected by validation loss (0.05891). The adapter and training record are under `outputs/qwen3-4b-arabic-qlora/`. Relative error reduction is `(base error - adapter error) / base error`; it was -41.67% for Arabic calls.

The accepted overlap audit is local at `reports/leakage_audit.json`. It checks exact and normalized prompts, tool names, full schemas, and expected calls, but cannot rule out semantic similarity.

## Local environment and manual training

Use Python 3.14, `uv`, and the repository `.venv`. The training kernel is named `Python (sila-4b .venv)`.

```powershell
uv sync --locked --group training
$env:PYTHONPATH = "src"
uv run python -m sila.validate_stress_set data/stress_test.jsonl --final
uv run python -m sila.audit_leakage
```

For an explicitly requested new training run, open the notebook with the `Python (sila-4b .venv)` kernel and run its cells from top to bottom. It reports the expected step count and live training progress. The full run takes hours on the local 8 GB RTX 4060. Preserve the completed experiment before writing another run to the same output directory. Do not rerun the corpus generator unless you intend to replace the current generated data.

The starting QLoRA configuration uses LoRA rank 16/alpha 32/dropout 0.05, an effective batch of 16, and three epochs (990 optimizer steps). The peak learning rate is `1e-4`, with 3% warmup (30 steps) followed by linear decay. Seed 42 is applied before adapter initialization; the lowest validation-loss checkpoint is selected. These settings have not been tuned or shown to meet the experiment's success criteria.

Development checks:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Data and reproducibility

### Evaluate the saved adapter

Restart or shut down the completed training notebook kernel first. Evaluation loads the cached pinned base once in NF4, attaches the saved adapter, and disables it for base inference. Both variants use identical native tool prompts and greedy decoding (512 new tokens maximum). No training or generated tool execution occurs.

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
uv run python -m sila.evaluate_models --prepare-metrics
uv run python -m sila.evaluate_models --suite all --limit 3 --output-dir outputs/evaluation/smoke
uv run python -m sila.evaluate_models
```

The smoke run checks the first three cases of each set and cannot establish the success criteria. Keep checkpoint and decoding settings fixed for the full evaluation. Repeat the same command to resume: each raw response is flushed immediately, and mismatched settings or inputs are rejected. Results stay ignored under `outputs/evaluation/`; `results/summary.json` contains counts, rates, and the verdict, and `results/arabic.paired.json` contains prompts, expected calls, and both raw responses. Use `--score-only` with the same suite, limit, and output directory to rescore complete saved predictions without loading model weights.

**Recorded-run caveat:** the base ArabFuncBench file retains 999 of 1,000 raw responses. Source row 837, `education_neg_4_008`, has a logged completion and metric record but no retained generated text; the cause is undetermined. The existing all-suite run cannot pass normal resume or score-only validation because resume requires a contiguous prefix. Preserve its files and reported scores; do not insert a new response without recording its provenance. Arabic and BFCL raw records are complete. Editing the runner, changing suite selection, or changing inputs also makes the existing run metadata incompatible; use a separate output directory for a different run.

Metric sources are downloaded separately into `outputs/evaluation/vendor/` and checked against pinned SHA-256 hashes before loading. BFCL scoring uses its original schemas and all possible answers. Its inference-client imports are omitted; a scoring-only registry preserves Qwen's original dot-containing function names. ArabFuncBench loads only upstream metric definitions and translates valid native calls into their input format. Malformed outputs fail negative cases rather than becoming null through the upstream JSON parser's fallback. Its TSA/AEF1/LCR are an external comparison under these native prompts, not a reproduction of upstream prompting. AEF1/LCR follow upstream conditional denominators. The export repeats ten IDs; internal row suffixes distinguish all 1,000 cases, while the data and upstream result IDs remain unchanged.

### Pinned inputs

- Base model: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Training uses bitsandbytes NF4 4-bit quantization with LoRA.
- ArabFuncBench: revision `c11e4e5ede503892b85633c40c16720c75310e48`; licensed CC BY 4.0. Keep it evaluation-only and use the [upstream metrics](https://github.com/lsadouk/ArabFuncBench) for its benchmark scores.
- BFCL V4: Gorilla revision `9d8416a96d1d69975493f1b6d60ff07d12a1726a`; Apache 2.0. Only the fixed subset described above is in scope.
- Training corpus: first-party generator v1.7. Its manifest pins the local data hashes and split details.
- JAIS (`inception42/jais-13b-chat`) is a planned comparator. Its immutable revision, license, and custom code still need review before download or use.

Local datasets, model outputs, and reports are ignored by Git. Preserve raw model responses for scoring, and never execute generated tool calls.

### Fresh-clone setup

Run from the repository root. Use `uv sync --locked --group training` to install the recorded dependency versions, then register the notebook kernel:

```powershell
uv run python -m ipykernel install --user --name sila-4b --display-name "Python (sila-4b .venv)"
$env:PYTHONPATH = "src"
```

**Existing checkouts must reuse their current training files and authoritative manifest.** On a fresh clone only, intentionally create the deterministic v1.7 corpus with:

```powershell
uv run python -m sila.generate_training_corpus --output-dir data
```

This writes `training.train.jsonl`, `training.validation.jsonl`, and `training.manifest.json`; it replaces those files if present. The original experiment's split hashes are:

| File | SHA-256 |
| --- | --- |
| `training.train.jsonl` | `dd7053cf0b6541313f59c57c4eb3a974b44fa22f7eba1ebfc0df765120b18bab` |
| `training.validation.jsonl` | `94eca3ed1bcde20e7cd9e6e1adde9f65991611b1ad4bb1fd3563dc2e274ca662` |

Obtain the pinned [ArabFuncBench dataset](https://huggingface.co/datasets/lsadouk1111/ArabFuncBench/tree/c11e4e5ede503892b85633c40c16720c75310e48). If access is gated, request it on the dataset page and authenticate with `uv run hf auth login` first:

```powershell
uv run hf download lsadouk1111/ArabFuncBench arab_func_bench_examples.json arab_func_bench_tools.json --repo-type dataset --revision c11e4e5ede503892b85633c40c16720c75310e48 --local-dir data/ArabFuncBench
```

Download the two BFCL categories and their original possible answers from the pinned Gorilla revision:

```powershell
$bfclUrl = "https://raw.githubusercontent.com/ShishirPatil/gorilla/9d8416a96d1d69975493f1b6d60ff07d12a1726a/berkeley-function-call-leaderboard/data"
New-Item -ItemType Directory -Force data/benchmarks/bfcl_v4/possible_answer | Out-Null
foreach ($category in @("simple_python", "multiple")) {
    $fileName = "BFCL_v4_$category.json"
    Invoke-WebRequest "$bfclUrl/$fileName" -OutFile "data/benchmarks/bfcl_v4/$fileName"
    Invoke-WebRequest "$bfclUrl/possible_answer/$fileName" -OutFile "data/benchmarks/bfcl_v4/possible_answer/$fileName"
}
```

The owner-reviewed `data/stress_test.jsonl` is local and is **not distributed in this repository**. Full reproduction requires an unchanged copy from the owner; otherwise the review and leakage gates cannot pass. `stress_test.template.jsonl` is illustrative and cannot replace the reviewed set. Do not relabel or rewrite it to bypass the gate. Once all inputs are available, run the validation and leakage-audit commands in this guide before opening the training notebook. Benchmark scoring also requires [ArabFuncBench's upstream metrics at the audited revision](https://github.com/lsadouk/ArabFuncBench/tree/9d314f34b6cb54d2916e4e2b454b87911df3110e) and BFCL's official AST evaluator at the pinned Gorilla revision.

### Evaluation output contract

The runner renders prompts with the pinned Qwen tokenizer's `apply_chat_template`, per-example function schemas, and `add_generation_prompt=True`, identically for base and adapter. Preserve the generated assistant text exactly in `RawPrediction.generated_text`; decode only the generated tokens and omit tokenizer special tokens. Native calls use `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`; no-call replies are ordinary assistant prose.

For the reviewed Arabic set, use `score_examples(examples, predictions, output_format="qwen")` for **both** models. This accepts one complete native call block (with optional surrounding prose), canonical JSON calls, JSON `null`, or non-empty plain prose as a no-call decision. Empty replies, malformed structured output, and multiple calls fail scoring. Plain prose does not count toward JSON parse success or valid structured output. The default `output_format="json"` retains the strict JSON-only contract for existing callers. Never select a format based on expected labels, repair malformed calls using gold answers, or execute generated calls. Use the upstream evaluators for benchmark scores rather than treating this internal scorer as BFCL or ArabFuncBench's official metric.
