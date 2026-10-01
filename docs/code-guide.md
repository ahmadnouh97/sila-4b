# Understanding the SILA-4B code

This guide explains the repository in reading order. You can read it without running training, downloading a model, or changing any data. For installation and commands, use the [reproduction guide](reproduction.md); for results, use the [README](../README.md).

## 1. What the project does

The question is: **does an Arabic-trained adapter improve tool calling compared with the original Qwen model?**

A tool call is a structured request such as “use the weather tool with city = Istanbul.” The model writes the request as text. This project checks that text against an expected answer; it never actually calls the weather service or any other generated tool.

The experiment has two paths:

```text
Preparing and training:
synthetic examples → validation/leakage checks → training notebook → saved adapter

Evaluating:
evaluation examples + base/adapter → raw answers → scoring → comparison verdict
```

Training teaches the adapter from examples that include correct answers. Evaluation gives the model only a user request and available tool definitions, then checks its response afterward. The evaluation answers must not enter the model prompt or training data.

### Terms you will see

| Term | Plain meaning |
| --- | --- |
| Base model | The original Qwen model, before this experiment's fine-tuning. |
| Adapter / LoRA | A small set of learned weights attached to the base. The original base weights remain frozen during training. |
| QLoRA / NF4 | Training LoRA while keeping the base weights in a compressed 4-bit representation to fit the GPU. Computation still uses a higher precision. |
| Inference | Asking a model to produce an answer, without updating weights. |
| Tokenizer | Converts text to model tokens and formats conversations/tool definitions. |
| Schema | A description of allowed fields and types, such as a required string named `city`. |
| Ground truth / expected answer | The answer used by the scorer, hidden from the model during evaluation. |
| Positive / negative case | A case that should / should not produce a tool call. “Negative” does not mean a bad example. |
| JSONL | A text file containing one JSON object per line. |
| Hash / SHA-256 | A file fingerprint used to detect changed input files or artifacts. |
| Pinned revision | An exact model or dataset version, rather than whichever version is newest. |
| AST | Python's representation of code structure. Here it helps load trusted benchmark scoring code without running whole upstream notebooks or clients. |

In Python, `def` defines a function; `class` defines a record/type. A name beginning with `_` is intended as an internal helper. `main()` reads command-line options and starts a module's workflow. The final `if __name__ == "__main__":` block runs `main()` when you launch that module, rather than when another file imports it.

## 2. `schemas.py`: the shared records

File: [src/sila/schemas.py](../src/sila/schemas.py)

Different datasets use different field names. The rest of the code works with these shared Python records:

| Record | What it holds |
| --- | --- |
| `ToolDefinition` | A tool's name, description, and parameter schema. |
| `EvaluationExample` | One user request, available tools, and the expected call/no-call answer. Also its ID, source, language, and category/domain. |
| `RawPrediction` | The model's original response text, example ID, model/settings information, time, and optional GPU memory measurement. |
| `ExpectedToolCall` | The correct tool name and arguments. |
| `ParsedToolCall` | The tool name and arguments extracted from the model's response. |
| `EvaluationResult` | Expected answer, raw answer, parsed answer, correctness flags, and any parsing error for one case. |

These are `dataclass` records: Python supplies their constructors and common record behavior. `frozen=True` prevents reassignment of their attributes, although dictionaries stored inside them are still mutable. `slots=True` restricts their instance attributes.

Each record's `__post_init__()` checks its contents immediately after construction. For example, a positive case must name a tool that is available; a no-call case must not have an expected tool name or arguments. A result and its raw prediction must refer to the same example ID.

The `_require_text()`, `_require_json()`, `_require_call()`, and `_require_non_negative_number()` helpers perform those basic checks. `_require_json()` rejects values that cannot be valid JSON, including non-finite numbers.

**Boundary:** these record checks are not a complete JSON Schema validator. The stricter reviewed-data checks live in `validate_stress_set.py`.

## 3. `evaluation.py`: score a response

File: [src/sila/evaluation.py](../src/sila/evaluation.py)

This file needs no GPU. It takes existing examples and existing response text, then measures correctness.

| Function / record | What it does |
| --- | --- |
| `_response_value(text, output_format)` | Parses JSON or, in `qwen` mode, one `<tool_call>...</tool_call>` block. Eligible non-empty plain prose means “no call” in Qwen mode. |
| `_no_duplicates()` | Rejects repeated JSON keys instead of silently keeping the last value. |
| `_reject_constant()` | Rejects non-standard JSON values such as `NaN`. |
| `_equal(left, right)` | Compares argument values recursively. `3` and `3.0` match; `true` and `1` do not; `"3"` and `3` do not. Dictionary order does not matter; list order does. |
| `score_examples(examples, predictions, output_format=...)` | Matches predictions to example IDs, parses each response, checks call/no-call choice, tool name, and exact arguments, then aggregates counts and rates. |
| `ScoredExample` | One result plus parsing flags and argument-key counts. |
| `ScoreReport` | All scored cases, total counts, and rates such as exact-call accuracy. |

`score_examples()` rejects missing, extra, or duplicate prediction IDs. A parsed call must contain exactly `name` and `arguments`. Empty answers, malformed structured answers, and multiple Qwen call blocks fail; they are not treated as successful refusals.

Exact-call accuracy uses only positive cases. No-call accuracy uses only negative cases. Argument-key precision/recall check which top-level field names were produced, not whether their values are right. A response can therefore have the right keys but the wrong city and still fail exact-call scoring.

Plain prose being accepted as no-call does not mean it is helpful, truthful, or safe. This scorer does not judge prose quality. It also does not execute a parsed call or provide full parameter-schema validation for model responses.

### A small example

This is an invented teaching example, not a case from the reviewed evaluation set:

```text
User: ما الطقس في إسطنبول؟
Available tool: weather(city: string)
Expected: weather with {"city": "Istanbul"}

Model response:
<tool_call>{"name":"weather","arguments":{"city":"Istanbul"}}</tool_call>
```

`_response_value()` extracts the JSON. `score_examples()` sees the correct tool and arguments, so exact match is true. If the model instead says `{"city":"Ankara"}`, tool selection is right but the arguments and exact match are wrong. If it produces plain prose, the no-call decision is wrong for this positive case.

## 4. `benchmarks.py`: translate external datasets

File: [src/sila/benchmarks.py](../src/sila/benchmarks.py)

This file reads already-downloaded benchmark data and converts it into `EvaluationExample` records. It does not download data, generate model answers, or score them.

| Function / record | What it does |
| --- | --- |
| `load_arabfuncbench(examples_path, tools_path, revision=...)` | Reads Arabic examples and tool definitions, resolves each example's tool names, and creates shared records. Repeated upstream IDs get internal row suffixes so every row stays distinct. |
| `load_bfcl(data_dir, revision=..., limit=500, categories=...)` | Reads the fixed BFCL category order, pairs questions with possible answers by ID, and returns a deterministic subset. It supports the project's single-call Python cases. |
| `BfclBatch` | Holds shared examples plus the original BFCL prompts and full ground truth. |
| `_jsonl()` | Reads a file's JSON objects line by line. Similar small readers appear in other modules. |

**Important BFCL detail:** some questions permit several correct values or optional arguments. The shared example contains one representative answer, but that is not enough to calculate the official score. `BfclBatch.ground_truth` preserves all alternatives for the official evaluator. The `multiple` category here means multiple tools are offered, not that this runner supports arbitrary multi-call conversations.

## 5. `benchmark_metrics.py`: use upstream scoring rules

File: [src/sila/benchmark_metrics.py](../src/sila/benchmark_metrics.py)

The internal exact-match scorer and external benchmark metrics are different. This module connects parsed responses to the benchmark authors' scoring functions.

| Function | What it does |
| --- | --- |
| `verified_source(vendor, name)` | Reads a metric source file and rejects it unless its hash matches `SOURCES`. |
| `prepare_metrics(vendor)` | Downloads the six pinned scoring source files into local ignored storage and checks their hashes. This uses the network; it does not download a model. |
| `load_bfcl_checker(vendor)` | Loads hash-verified official scoring code while omitting its BFCL package imports. Supplies the small naming registry needed for Qwen's original function names. |
| `load_arab_metrics(vendor, tools)` | Extracts only four upstream function definitions: `char_similarity`, `is_arabic`, `evaluate_response`, and `compute_metrics`. It does not execute the full upstream notebook. |
| `score_bfcl(batch, report, namespace)` | Sends parsed calls, original schemas, and all accepted answers to BFCL's AST checker. Returns per-case results and official accuracy. |
| `score_arabfuncbench(rows, report, namespace)` | Converts parsed calls to the upstream input format and computes its metrics. Pairs repeated IDs by source-row order; malformed output remains a failure even on negative cases. |

`namespace` is a dictionary containing loaded scoring functions and their supporting names. You may notice `exec(compile(...))`: it runs the pinned, hash-checked evaluator source. **It does not run the model's generated text.** Model answers are parsed as data.

ArabFuncBench's TSA measures tool selection; AEF1 and LCR are conditional upstream metrics. Improvements in those conditional metrics do not establish improved exact-call accuracy over the whole set.

## 6. `evaluate_models.py`: ask both models, save, and compare

File: [src/sila/evaluate_models.py](../src/sila/evaluate_models.py)

This is the evaluation coordinator. It connects the data loaders, model, response scorer, and benchmark scorers.

| Function | What it does |
| --- | --- |
| `load_inputs(root, run_dir)` | Loads the training record and leakage audit; verifies pinned revisions, data hashes, review status, and fixed dataset sizes; loads all three evaluation sets. |
| `render_prompt(tokenizer, example)` | Formats the user's text and available tools with Qwen's chat template. It excludes expected answers. |
| `ensure_run_record(path, settings)` | Creates a run-settings record or refuses to continue if the existing record differs. Prevents mixing incompatible runs. |
| `read_predictions(path, examples, variant, generation)` | Loads saved raw responses and checks that they are the correct model/settings and a contiguous prefix of the expected examples. |
| `generate_predictions(...)` | Generates only the remaining answers, records time/memory and prompt hashes, and appends each raw response to JSONL. Flushes and requests a disk sync after every answer. |
| `summarize(...)` | Runs internal parsing/scoring and the appropriate external scorer, writes per-model scores and Arabic paired cases, and saves `summary.json`. |
| `comparison(...)` | Applies the three preset criteria and reports score changes and success/failure. An incomplete primary comparison cannot receive a final verdict. |
| `write_json(path, value)` | Writes to a temporary file, then replaces the target, reducing the risk of leaving a partially written JSON report. |
| `sha256(path)` | Computes a file fingerprint. |
| `main()` | Reads options, validates inputs/settings, loads the GPU model only when generation is needed, runs generation, and produces the summary. |

The base and adapter do not require two separately loaded base models. `main()` loads the pinned NF4 base once and attaches the saved adapter with `is_trainable=False`. For the base pass, `generate_predictions()` uses `model.disable_adapter()`; for the adapter pass, it leaves the adapter active. Both passes use the same prompt renderer and decoding settings. The current loop completes a variant's cases before switching, rather than alternating models after every question.

`model.eval()` selects evaluation behavior, and `torch.inference_mode()` avoids gradient tracking. There is no training call in this module.

Useful options to understand, without launching them:

- `--prepare-metrics`: download/check evaluator sources, then exit.
- `--limit 3`: a small smoke run, not a final performance result.
- `--score-only`: score complete compatible saved predictions without loading model weights. Local inputs and metric sources are still required.
- `--output-dir`: choose where a run's metadata, raw predictions, and scores go.

Resume has a real limit: it can continue after a valid saved prefix, but it rejects a broken final JSON line or a missing middle response. It does not silently repair either. The recorded ArabFuncBench base file has a missing middle response; its caveat remains in the README.

## 7. `validate_stress_set.py`: check reviewed evaluation data

File: [src/sila/validate_stress_set.py](../src/sila/validate_stress_set.py)

| Function | What it does |
| --- | --- |
| `_check_schema(schema)` | Checks supported schema types, required fields, enums, date/time formats, minimums, and restrictions on extra properties. |
| `_check_value(value, schema, field)` | Checks an expected answer against that schema, including nested values and required fields. |
| `validate_stress_set(path, final=False, pilot=False, draft=False)` | Checks every row, unique IDs/prompts, Arabic text, categories, tool consistency, review labels, reference dates, expected answers, and category counts. Returns category counts or raises an error with the row number. |
| `main()` | Selects a file and validation mode from command-line options and reports whether it passed. |

The modes require different counts: template = 1 per category, pilot = 3, draft = 10, final = 10. Final also requires reviewed status and no remaining review notes. Recognized relative-date requests must include the fixed reference date in the prompt.

This checks structure and declared review status. It cannot verify that an Arabic sentence is natural or that a human actually reviewed it. The owner's review provides that part.

## 8. `audit_leakage.py`: check training/evaluation overlap

File: [src/sila/audit_leakage.py](../src/sila/audit_leakage.py)

Leakage means evaluation information appears in training, making a score look better without demonstrating generalization.

| Function | What it does |
| --- | --- |
| `_normalize_prompt(text)` | Normalizes Unicode and case, replaces marks/punctuation/symbols with spaces, and collapses whitespace to catch some differently written duplicates. |
| `_canonical(value)` | Produces consistently ordered JSON text so dictionaries can be compared as stable strings. |
| `_tool_parts()` / `_tool_sets()` | Extract tool names, complete schemas, and parameter names for comparisons. |
| `_target_set(rows)` | Collects expected tool-name/argument pairs from positive training examples. |
| `_matches_bfcl_target(tool_name, arguments, answer)` | Checks overlap against BFCL's allowed alternatives and optional arguments. |
| `_audit_group(...)` | Compares one evaluation set with the corpus: exact/normalized prompts, tool names, full schemas, and expected calls. Reports parameter-name overlap as a diagnostic, not a blocker. |
| `run_audit()` | Reads train plus diagnostic validation data and all evaluation sets, audits each group, and writes `reports/leakage_audit.json` with hashes and pass/block status. |
| `main()` | Runs the audit and prints its status. |

The audit uses both training and diagnostic validation rows. It catches defined string/structure overlaps; it does not understand meaning and cannot exclude semantic similarity.

## 9. `generate_training_corpus.py`: build synthetic teaching examples

File: [src/sila/generate_training_corpus.py](../src/sila/generate_training_corpus.py)

Most of this file is **data**, not algorithm: Arabic sentence templates, tool descriptions, example values, dialect forms, and no-call answers. It uses Python templates and combinations, not an external LLM API.

`TASKS` defines ten positive tool tasks across the corpus. `VALIDATION_TOOLS` holds out two whole tools for diagnostic validation, leaving eight training tools. `PROMPT_FORMS` contains wording variations for six dialect labels; the `NO_CALL_*` collections define requests that should get prose answers instead of tool calls.

| Function | What it does |
| --- | --- |
| `_tool(task)` | Builds a tool definition with parameter names/types, required fields, and no extra properties. |
| `_prompt_value(field, value, form_index)` | Changes how a value appears in a prompt, such as Arabic date wording or hamza spelling, while keeping the expected target value. |
| `_value_combinations(task)` | Builds value combinations, keeps linked fields together, and deterministically selects 100 combinations per tool. |
| `generate_rows()` | Creates positive tool-call conversations and negative prose conversations, assigns train/validation splits, and varies competing tools and their order. Returns rows in memory. |
| `_write_jsonl(path, rows)` | Writes rows as one JSON object per line and returns the written bytes for hashing. |
| `generate(output_dir)` | Checks counts, uniqueness, targets, and variation/balance rules; writes both splits and their manifest. Returns the manifest. |
| `main()` | Selects the output directory and calls `generate()`. |

The full corpus is 6,600 rows: 6,000 calls and 600 no-call examples. The split is 5,280 training and 1,320 diagnostic validation rows. Positive validation holds out whole tools; negative validation holds out forms/content. This is not a random row split.

The manifest records versions, counts, split policy, and file hashes. Synthetic dialect labels do not establish linguistic quality; training text has not had linguistic review. **Running `generate()` replaces the existing output files. Preserve the authoritative corpus unless regeneration is explicitly requested.**

## 10. The training notebook: the only training workflow

File: [notebooks/train_qwen_qlora.ipynb](../notebooks/train_qwen_qlora.ipynb)

Notebook cells share variables from earlier cells. Read them top to bottom; reading them does not require executing them.

| Stage | Significant code and purpose |
| --- | --- |
| Paths and base version | Sets `ROOT`, data/output paths, `MODEL_ID`, and `MODEL_REVISION`. |
| Input gates | `require()` stops on a failed condition; `sha256()` checks fingerprints. Cells verify the manifest, review metadata, and required leakage-audit fields before loading model weights. |
| Dataset conversion | `read_jsonl()` reads rows. `to_trl_example()` keeps their conversations and wraps tool definitions in TRL's expected format. |
| GPU and token lengths | Chooses BF16/FP16 compute. `sequence_length()` measures tokenized conversations so oversized examples raise an error. |
| Base loading | `BitsAndBytesConfig` specifies NF4 double quantization. `AutoModelForCausalLM.from_pretrained()` loads the pinned base; `prepare_model_for_kbit_training()` prepares it for adapter training. |
| Adapter configuration | `LoraConfig` sets rank 16, alpha 32, dropout 0.05, and attention/MLP target projections. |
| Training settings | The prepared v1.8 `SFTConfig` caps training at 330 updates, saves at 110/220/330, and keeps assistant-only loss. Development generations select a candidate; validation loss is diagnostic. |
| Trainer creation | `SFTTrainer` connects model, data, tokenizer, and adapter configuration. `set_seed()` is called before adapter initialization. |
| Training and saving | **`trainer.train()` starts training.** `trainer.save_model()` saves the adapter, the tokenizer is saved, and `run.json` records settings, hashes, and training metrics. |

The completed v1.7 run used batch 4 with accumulation 4 and three epochs (990 updates). The prepared v1.8 trial uses batch 2 with accumulation 8 (still effective batch 16), learning rate `5e-5`, and at most 330 updates. Its longer rendered examples use a 768-token limit. Assistant-only loss supervises assistant output rather than user/schema tokens. Gradient checkpointing reduces memory use. Training remains disabled until owner authorization.

The selected adapter came from epoch one, despite training for three epochs. Selection used diagnostic validation loss, not the final benchmark scores. Low validation loss does not guarantee better tool calling.

The new [follow-up generator](../src/sila/followup_data.py) creates separate data and held-out development cases; [development scoring](../src/sila/development.py) reuses native prompting, raw-response persistence, and the exact-call scorer. It selects only hash-verified checkpoints with Arabic gains and no English/no-call regression. This first-party diagnostic is separate from official final BFCL scoring.

**Do not run the training cell just to explore the code.** The experiment is already complete.

## 11. Tests and supporting files

Tests use small examples and temporary files. `tmp_path` is pytest's temporary directory; `pytest.raises(...)` checks that bad input is rejected; an `assert` checks an expected value. Parameterized tests repeat the same check for several inputs, which is why test count exceeds the number of named test functions.

| File | Purpose and useful checks to read |
| --- | --- |
| [tests/test_schemas.py](../tests/test_schemas.py) | Valid records and invalid/inconsistent records, including preserving raw text. Read `test_valid_records_preserve_raw_prediction()`. |
| [tests/test_evaluation.py](../tests/test_evaluation.py) | Hand-calculated scores, value comparisons, ID matching, duplicate JSON keys, and native Qwen parsing. Start with `test_hand_calculated_fixture_metrics()` and `test_qwen_native_response_contract()`. |
| [tests/test_benchmarks.py](../tests/test_benchmarks.py) | Small local benchmark fixtures; verifies loading and preserving BFCL's full accepted answers. |
| [tests/test_stress_set.py](../tests/test_stress_set.py) | Template validity and rejection of bad schemas, labels, dates, counts, and duplicate rows. |
| [tests/test_evaluate_models.py](../tests/test_evaluate_models.py) | Hidden gold answers, resume compatibility, all-three acceptance criteria, repeated benchmark IDs, malformed responses, and modified evaluator-source rejection. These do not run a real GPU model. |
| [pyproject.toml](../pyproject.toml) | Python requirement, dependencies, CUDA package source, and pytest/ruff settings. |
| [uv.lock](../uv.lock) | Exact resolved dependency versions. It is generated dependency metadata, not application logic to study line by line. |
| [.gitignore](../.gitignore) | Keeps virtual environments, local datasets, adapters, predictions, and reports out of Git. |
| [AGENTS.md](../AGENTS.md) | Repository instructions, including preserving evaluation data and not launching training without a request. |
| [TODO.md](../TODO.md) | Completed work and possible follow-ups. |
| [README.md](../README.md) | Experiment design, measured results, and limitations. |
| [docs/reproduction.md](reproduction.md) | Environment, input setup, and workflow commands. |
| `docs/portfolio.md` (local only) | Résumé/portfolio wording; deliberately uncommitted. It has no runtime role. |
| [data/stress_test.template.jsonl](../data/stress_test.template.jsonl) | Ten illustrative cases for understanding/validating the format. It is not the reviewed 100-case set. |
| `.gitkeep` files | Empty placeholders so otherwise empty data/output/report folders can exist in Git. |

There is no automated test of a real training run here. These tests protect specified data/scoring behaviors; they cannot establish model performance, Arabic fluency, or general safety.

### Local artifacts you may see

| Path | Meaning |
| --- | --- |
| `data/training.train.jsonl` / `training.validation.jsonl` | Synthetic teaching examples and diagnostic validation examples. |
| `data/training.manifest.json` | Authoritative description and fingerprints of that corpus. |
| `data/stress_test.jsonl` | Owner-reviewed evaluation-only Arabic cases. |
| `data/ArabFuncBench/` / `data/benchmarks/bfcl_v4/` | Local external evaluation inputs. |
| `outputs/qwen3-4b-arabic-qlora/adapter/` | Learned adapter weights/configuration and saved tokenizer files. |
| `outputs/qwen3-4b-arabic-qlora/run.json` | Training provenance and metrics. |
| `outputs/evaluation/results/*.raw.jsonl` | Original model responses and inference metadata; preserve these for rescoring. |
| `outputs/evaluation/results/*.scores.json` / `summary.json` | Derived per-case scores and overall comparison. |
| `outputs/evaluation/results/arabic.paired.json` | Arabic expected answers plus base and adapter responses/results, side by side. |
| `outputs/evaluation/results/run.json` | Evaluation settings and fingerprints; different from the training `run.json`. |
| `outputs/evaluation/vendor/` | Pinned upstream scorer sources. |
| `reports/leakage_audit.json` / `evaluation_review.md` | Input-overlap audit and human-readable result review. |
| `.venv/` and cache folders | Installed dependencies and temporary tool files. |
| `outputs/github-profile/` | Local checkout used for the profile edit; unrelated to model training or scoring. |

These are local/ignored artifacts, so a fresh clone does not contain the recorded experiment's full evidence. Scores are derived from raw responses; they are not interchangeable with the original responses.

## 12. A practical reading order

1. Read the example in section 3 and inspect `EvaluationExample`, `RawPrediction`, and `EvaluationResult`.
2. Read `score_examples()` alongside its hand-calculated test. Understand what “correct” means before studying model loading.
3. Read `render_prompt()`, `generate_predictions()`, and `summarize()` to follow one response from prompt to report.
4. Read the benchmark loaders/scorers to understand why the official BFCL result differs from internal exact matching.
5. Read the notebook's `to_trl_example()`, `LoraConfig`, `SFTConfig`, and training/saving cell without executing them.
6. Finish with generator templates, `generate_rows()`, validation, and the leakage audit.

You understand the core flow when you can explain where the correct answers live, why they are hidden during inference, what changes during LoRA training, how the base/adapter passes differ, and why the saved adapter failed the acceptance criteria.
