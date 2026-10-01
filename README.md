# SILA-4B

SILA-4B tests whether a small Arabic-focused QLoRA adapter improves tool calling while preserving the base model's English performance.

## Current status

Training completed on September 30, 2026: three epochs, 990 optimizer steps, and 17,921.7 seconds (about 4 hours 59 minutes). The saved adapter matches `checkpoint-330` (epoch 1), selected by validation loss 0.05891. Paired evaluation finished on October 1. **The adapter failed the experiment's success criteria:** Arabic exact-call accuracy fell by 7.14 percentage points, English BFCL fell by 2.40 points, and Arabic no-call accuracy was unchanged. Outputs are local under `outputs/evaluation/results/`; the review is `reports/evaluation_review.md`. One ArabFuncBench base raw response is missing despite a logged completion and metric record; its external scores have a reproducibility caveat.

## Experiment and success criteria

Compare the untouched base model with its Arabic-only adapter using the same prompts, tools, decoding settings, and quantization. Training uses 5,280 synthetic Arabic examples; 1,320 held-out examples are used for training diagnostics. The evaluation sets are never used for training.

The experiment succeeds only if the adapter meets all three criteria:

1. Arabic exact tool-call accuracy improves by **at least 5 percentage points**, or relative error falls by **at least 25%**.
2. English BFCL accuracy falls by **no more than 2 percentage points**.
3. Arabic no-call accuracy does not decrease.

Relative error reduction is `(base error - adapter error) / base error`.

The primary Arabic evaluation is the owner-reviewed 100-case set. ArabFuncBench (1,000 cases) provides an external Arabic comparison. BFCL V4 uses a fixed 500-case English subset: 400 `simple_python` and 100 `multiple`. Score BFCL with its [official AST evaluator](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py) and original records.

The local leakage audit checks exact and normalized prompts, tool names, full schemas, and expected calls; it cannot rule out semantic similarity. The audit report is `reports/leakage_audit.json` and is ignored by Git along with local data and outputs.

## Results

**Overall verdict: failed.** The reviewed Arabic and BFCL results reproduce from complete saved raw responses with identical prompts, NF4 quantization, and greedy decoding.

| Evaluation | Base | Adapter | Change | Criterion met? |
| --- | --- | --- | --- | --- |
| Reviewed Arabic exact tool-call accuracy (70 call cases) | 58/70 (82.86%) | 53/70 (75.71%) | -7.14 pp | No |
| Reviewed Arabic no-call accuracy (30 no-call cases) | 17/30 (56.67%) | 17/30 (56.67%) | 0.00 pp | Yes |
| English BFCL official AST accuracy (fixed 500 cases) | 469/500 (93.80%) | 457/500 (91.40%) | -2.40 pp | No |
| ArabFuncBench upstream TSA (reported 1,000 cases) | 957/1,000 (95.70%) | 949/1,000 (94.90%) | -0.80 pp | External; raw caveat below |

Arabic call errors rose from 12 to 17: relative error reduction was **-41.67%**. Four lost call cases changed `dog` to `cat`; one unnecessarily asked for clarification. English had 18 regressions and six improvements, including regressions in percentages and currency-code normalization. Neither model handled Arabic prompt-injection no-call cases well (base 2/10, adapter 1/10).

ArabFuncBench's reported conditional AEF1 improved from 77.97% to 89.14%, and LCR from 84.84% to 95.28%; these are upstream conditional metrics, not exact-call accuracy over all cases. Base raw output has 999 records: `education_neg_4_008` is missing although the log and its original metric record exist. Preserve the original results; recover or regenerate that one case with provenance before claiming full external reproducibility. The primary verdict does not depend on this external set.

Inference used the RTX 4060 Laptop GPU, NF4 double quantization with BF16 compute, greedy decoding, and a 512-new-token limit. Recorded generation time totaled 9.225 hours, excluding loading and other overhead; peak allocated inference VRAM was 3,151.1 MiB (about 3.08 GiB). Training peak VRAM was not recorded. Prompts and decoding match between variants; adapter and training-record hashes are unchanged. Representative raw examples and the input audit are in the local review report.

Next: retain the base as the reference and preserve this failed run. Audit synthetic data quality and add coverage for enums, normalized numeric/date/code values, and grounded no-call decisions using independent development examples. The current training set has eight tool schemas, no enums or date formats, and only string/integer fields; this is a plausible coverage gap, not proof of the regressions' cause. Do not copy evaluation examples into training, tune checkpoints on these scores, regenerate the current manifest, or launch another training run without an owner request.

Limits: the reviewed Arabic set is small (10 cases per category), training data is synthetic and has not had linguistic review, the leakage audit does not check semantic similarity, and the BFCL subset does not establish preservation of general English ability. No-call accuracy measures the decision to avoid a tool call, not the quality or safety of the prose reply. A negative experiment result is still a result; report it without claiming improvement.

## Code guide

| File | What it does |
| --- | --- |
| `src/sila/generate_training_corpus.py` | Creates Arabic training examples, separates training and check sets, and records file hashes. Regenerating replaces those files. |
| `src/sila/validate_stress_set.py` | Checks required fields, categories, tool input descriptions, and review status in the Arabic evaluation file. It cannot judge translation quality. |
| `src/sila/audit_leakage.py` | Checks whether training examples repeat evaluation prompts, tools, or answers; writes the audit report. |
| `src/sila/benchmarks.py` | Reads local ArabFuncBench and BFCL files into the project's common format. It does not download data or score model outputs. |
| `src/sila/evaluation.py` | Scores canonical JSON or native Qwen responses against expected tool calls. It never runs a tool call. |
| `src/sila/evaluate_models.py` | Runs resumable paired NF4 inference, saves raw responses, scores all three sets, and reports the success criteria. |
| `src/sila/benchmark_metrics.py` | Loads hash-verified BFCL AST scoring and ArabFuncBench metric functions at the pinned upstream revisions. |
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

The starting QLoRA configuration uses LoRA rank 16/alpha 32/dropout 0.05, an effective batch of 16, and three epochs (990 optimizer steps). The peak learning rate is `1e-4`, with 3% warmup (30 steps) followed by linear decay. Seed 42 is applied before adapter initialization; the lowest validation-loss checkpoint is selected. These settings have not been tuned or shown to meet the experiment's success criteria.

Development checks:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format .
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

The owner-reviewed `data/stress_test.jsonl` is local and is **not distributed in this repository**. Full reproduction requires an unchanged copy from the owner; otherwise the review and leakage gates cannot pass. `stress_test.template.jsonl` is illustrative and cannot replace the reviewed set. Do not relabel or rewrite it to bypass the gate. Once all inputs are available, run the validation and leakage-audit commands above before opening the training notebook. Benchmark scoring also requires [ArabFuncBench's upstream metrics at the audited revision](https://github.com/lsadouk/ArabFuncBench/tree/9d314f34b6cb54d2916e4e2b454b87911df3110e) and BFCL's official AST evaluator at the pinned Gorilla revision.

### Evaluation output contract

The runner renders prompts with the pinned Qwen tokenizer's `apply_chat_template`, per-example function schemas, and `add_generation_prompt=True`, identically for base and adapter. Preserve the generated assistant text exactly in `RawPrediction.generated_text`; decode only the generated tokens and omit tokenizer special tokens. Native calls use `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`; no-call replies are ordinary assistant prose.

For the reviewed Arabic set, use `score_examples(examples, predictions, output_format="qwen")` for **both** models. This accepts one complete native call block (with optional surrounding prose), canonical JSON calls, JSON `null`, or non-empty plain prose as a no-call decision. Empty replies, malformed structured output, and multiple calls fail scoring. Plain prose does not count toward JSON parse success or valid structured output. The default `output_format="json"` retains the strict JSON-only contract for existing callers. Never select a format based on expected labels, repair malformed calls using gold answers, or execute generated calls. Use the upstream evaluators for benchmark scores rather than treating this internal scorer as BFCL or ArabFuncBench's official metric.
