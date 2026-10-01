# SILA-4B

An Arabic tool-calling experiment with Qwen3-4B, QLoRA, and paired evaluation on a single 8 GB laptop GPU.

**Result: the adapter failed the preset acceptance criteria.** Arabic exact-call accuracy fell by 7.14 percentage points, English BFCL accuracy fell by 2.40 points, and Arabic no-call accuracy stayed unchanged. The project records the training pipeline, evaluation controls, failure analysis, and limits of that result.

[Reproduction guide](docs/reproduction.md)

New to the code? Read the [plain-language code guide](docs/code-guide.md) for the file map, main functions, and a worked scoring example.

## Question and approach

Can Arabic-only fine-tuning improve tool calling while preserving English tool-calling performance?

- Fine-tune the pinned `Qwen/Qwen3-4B-Instruct-2507` base with bitsandbytes NF4 4-bit quantization and LoRA.
- Generate 5,280 first-party synthetic Arabic training examples and hold out 1,320 for training diagnostics. Keep all evaluation data separate.
- Compare base and adapter with identical tool schemas, native Qwen prompts, quantization, and greedy decoding. Preserve raw responses and never execute generated tool calls.
- Score a 100-case owner-reviewed Arabic set, a fixed 500-case English BFCL V4 subset, and 1,000 external ArabFuncBench cases. Use BFCL's official AST evaluator and ArabFuncBench's upstream metric functions at pinned revisions.
- Check exact and normalized overlap before training; pin model and dataset revisions, input hashes, and evaluation settings.

Acceptance requires **all three**: Arabic exact-call accuracy gains at least 5 percentage points or relative error falls by at least 25%; English BFCL loses no more than 2 points; Arabic no-call accuracy does not decrease.

## Results

Training completed September 30, 2026; evaluation finished October 1. The saved adapter matches the epoch-one checkpoint selected by validation loss, after a three-epoch run.

| Primary evaluation | Base | Adapter | Change | Pass |
| --- | --- | --- | --- | --- |
| Arabic exact calls, 70 call cases | 58/70 (82.86%) | 53/70 (75.71%) | -7.14 pp | No |
| Arabic no-call decisions, 30 cases | 17/30 (56.67%) | 17/30 (56.67%) | 0.00 pp | Yes |
| English BFCL official AST, 500 cases | 469/500 (93.80%) | 457/500 (91.40%) | -2.40 pp | No |

Arabic call errors rose from 12 to 17, a 41.67% increase. Tool-name accuracy stayed at 66/70; the losses came from argument values and an unnecessary clarification. English had 18 regressions and six improvements, including regressions in percentage and currency-code normalization. Arabic prompt-injection no-call accuracy was low for both models: 2/10 for the base and 1/10 for the adapter.

External ArabFuncBench reported tool-selection accuracy (TSA) of 95.70% → 94.90%. Conditional AEF1 rose from 77.97% → 89.14% and LCR from 84.84% → 95.28%; their conditional denominators make them different from overall exact-call accuracy. **One base raw response is missing**, despite a logged completion and retained metric record, so these external scores cannot be fully reconstructed from the retained raw files. The primary Arabic and BFCL records are complete and support the verdict independently.

| Run detail | Recorded value |
| --- | --- |
| Hardware | NVIDIA RTX 4060 Laptop GPU, 8 GB |
| LoRA | Rank 16, alpha 32, dropout 0.05; attention and MLP projections |
| Training | 3 epochs, 990 optimizer steps; effective batch 16, peak learning rate `1e-4`, seed 42 |
| Selected checkpoint | Epoch 1 (`checkpoint-330`), validation loss 0.05891 |
| Training runtime | About 4 hours 59 minutes |
| Paired generation time | 9.225 hours, excluding loading and other overhead |
| Peak allocated inference VRAM | 3,151.1 MiB (about 3.08 GiB); training peak was not recorded |
| Decoding | Greedy, maximum 512 new tokens, NF4 double quantization with BF16 compute |

## What the experiment established

The adapter should not replace the base for the measured tasks. Lower training loss and better conditional external metrics did not translate into better exact calls on the primary evaluation.

The synthetic corpus covers only eight tool schemas with string/integer fields, without enums or date formats. That is a plausible coverage gap to investigate, not a proven cause of the regressions. Any follow-up should first review data quality and test normalization and grounded no-call decisions on separate development examples. More training alone has no demonstrated benefit here.

The [next-experiment checklist](TODO.md) now has a separate v1.8 corpus (9,060 training / 2,328 validation rows), 276 independent development cases, and checkpoint scoring with English/no-call preservation gates. Structural/leakage checks and independent AI review pass; this is not native-speaker certification. The frozen development base scores 57/66 Arabic calls, 62/66 English calls, and 54/72 Arabic no-call decisions. An initial call ceiling prompted one fixed reviewed challenge appendix; both versions are preserved. The notebook is prepared for a bounded Qwen-only follow-up with training disabled; another run cannot guarantee an improvement.

The Arabic set is small, with ten cases per category; synthetic training text has not had linguistic review. The leakage audit cannot exclude semantic similarity, BFCL covers a fixed subset rather than general English ability, and no-call scoring measures a decision rather than reply quality or safety. This is one completed experiment, not evidence that QLoRA generally harms Arabic performance.

## Review the code without a GPU

Use Python 3.14 and [uv](https://docs.astral.sh/uv/). From a fresh clone:

```powershell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
$env:PYTHONPATH = "src"
uv run python -m sila.validate_stress_set data/stress_test.template.jsonl
```

The tests and illustrative template require no GPU, model download, or private evaluation data. The template is not the reviewed evaluation set and cannot establish model performance. For CUDA dependencies, pinned inputs, manual training, and saved-adapter evaluation, see the [reproduction guide](docs/reproduction.md).

| Entry point | Purpose |
| --- | --- |
| [Training notebook](notebooks/train_qwen_qlora.ipynb) | Follow-up review/hash gates, bounded NF4 QLoRA, checkpoints and run-record export; training requires authorization |
| [Original corpus generator](src/sila/generate_training_corpus.py) / [follow-up data](src/sila/followup_data.py) | Deterministic v1.7 reproduction and separate v1.8 additions/development cases |
| [Development scoring](src/sila/development.py) | Cached-base/checkpoint generation, exact-call scoring, and hash-verified candidate selection |
| [Validation](src/sila/validate_stress_set.py) / [leakage audit](src/sila/audit_leakage.py) | Structural/review checks and exact/normalized overlap checks |
| [Paired runner](src/sila/evaluate_models.py) | Raw-response persistence, compatible-prefix resume, inference, scoring, and acceptance verdict |
| [Benchmark loaders](src/sila/benchmarks.py) / [upstream metrics](src/sila/benchmark_metrics.py) | Pinned benchmark inputs, duplicate-row handling, hash-verified metric code |
| [Response scoring](src/sila/evaluation.py) / [schemas](src/sila/schemas.py) | Native-call parsing, schema validation, exact-call and no-call scoring |
| [Tests](tests) | Deterministic checks for parsing, validation, benchmark loading, resume metadata, and verdict logic |

## Evidence and availability

This repository contains source, tests, the illustrative template, and aggregate findings. The reviewed evaluation set, benchmark exports, adapter, raw predictions, and full reports remain local and ignored by Git. A fresh clone can exercise the code but cannot independently reproduce the reported scores without those artifacts. Local records are `outputs/qwen3-4b-arabic-qlora/run.json`, `outputs/evaluation/results/summary.json`, and `reports/evaluation_review.md`.

Existing checkouts must preserve the authoritative training manifest and reviewed evaluation data. Training is manual and must not be launched without an explicit owner request. Model and benchmark sources, revisions, attribution, and setup details are in the [reproduction guide](docs/reproduction.md).
