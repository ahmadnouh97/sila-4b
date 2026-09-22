# Sila-4B

Sila-4B is a reproducible Arabic tool-calling fine-tuning and evaluation project.

## Research question

Can Arabic-specific QLoRA improve tool selection and argument extraction on unseen Arabic tools without materially degrading English tool-calling ability?

## Experimental design

- **Untouched base model:** the frozen, revision-pinned 4B instruction checkpoint used to initialize the treatment. It is evaluated without adapters or weight updates. Its exact model ID and immutable revision must be recorded before any download.
- **QLoRA treatment:** an adapter trained only on Arabic tool-calling examples, with the base weights frozen. Training examples and tool schemas must be disjoint from every evaluation set.
- **Arabic unseen-tool set:** Arabic requests paired with held-out mock tool schemas, testing tool selection and argument extraction on tools absent from training.
- **English BFCL set:** a fixed, revision-pinned BFCL subset measuring whether English tool-calling ability regresses.
- **Arabic negative set:** Arabic requests for which no tool should be called, measuring false tool invocation.

The base and treatment use identical prompts, schemas, decoding settings, and quantization. Every raw response is saved before parsing. Generated calls are compared structurally with expected calls and are never executed.

## Metrics

The primary metric is exact tool-call accuracy on the Arabic unseen-tool set: the selected tool and normalized arguments must both match.

Secondary metrics are tool-selection accuracy, argument exact match, argument field F1, English BFCL accuracy, Arabic negative-case accuracy, invalid-response rate, and absolute and relative changes from the base model. Relative error reduction is `(base error - treatment error) / base error`.

## Leakage policy

Evaluation prompts, expected answers, tool schemas, paraphrases, and derived examples must never appear in training or tuning data. Dataset splits are fixed before training, checked for exact duplicates, and pinned to immutable revisions. Any contaminated example invalidates the affected comparison.

## Manually authored Arabic stress set

`data/stress_test.template.jsonl` contains **10 illustrative examples, not gold labels**: one each for direct requests, dialect requests, code switching, spelling noise, distractor tools, multiple arguments, argument normalization, missing required input, out-of-scope requests, and prompt injection. Each JSONL row has `id`, `category`, `dialect`, `review_status`, `user_utterance`, `available_tools`, `should_call_tool`, `expected_tool_name`, and `expected_arguments`. Negative rows use JSON `null` for both expected fields. Allowed dialect labels are `msa`, `egyptian`, `levantine`, `gulf`, `iraqi`, and `maghrebi`.

The final evaluation set requires **100 manually authored cases, 10 per category**, one fixed `reference_date` across cases, and manual native-speaker review of every prompt, tool schema, and expected answer. Set `review_status` to `native_speaker_reviewed` only after that review and resolve every `review_note`. Generated draft examples must not be treated as gold labels. Keep the final set untracked under `data/` and disjoint from training and tuning data.

Run `uv run python -m sila.validate_stress_set data/stress_test.template.jsonl` for the illustrative template, `uv run python -m sila.validate_stress_set data/stress_test.jsonl --draft` for the unreviewed 100-case set, or the same command with `--final` after every row has been reviewed. The validator checks structure and counts; it cannot establish linguistic quality or prove that review took place.

The untracked `data/stress_test.jsonl` is a **100-case draft**, with ten cases per category, ten fictional mock tools defined inline, and `reference_date: "2026-09-22"` for relative dates. Relative-date utterances include that date in their text, so it reaches prompts built from `user_utterance`. Every positive case offers at least two tools. The six flagged labels in the original 30-case pilot were confirmed by the user. All 100 rows have had an AI editorial pass, but the added cases have not had manual native-speaker review. All labels remain draft until native-speaker review and dataset freeze. Do not run either model on this set until then. Before freezing, compare tool names, schemas, and prompts against the actual revision-pinned training and external benchmark data; those sources are not available in this repository.

## External benchmark adapters

`sila.benchmarks.load_arabfuncbench(examples_path, tools_path, revision=...)` reads the two local JSON exports from [ArabFuncBench](https://huggingface.co/datasets/lsadouk1111/ArabFuncBench). The dataset is gated and licensed **CC BY 4.0**. It remains evaluation-only; do not put any of its prompts, tools, labels, or derivatives into training. The upstream [evaluation notebooks](https://github.com/lsadouk/ArabFuncBench) define tool selection accuracy, fuzzy argument extraction F1, and language compliance. Our existing exact-match scorer is a separate metric and must not be reported as the upstream benchmark score.

`sila.benchmarks.load_bfcl(data_dir, revision=..., limit=500)` reads local BFCL V4 JSONL files and matching `possible_answer/` files from the [official Gorilla repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data). BFCL data and code are **Apache 2.0**. The default fixed prefix is 400 `simple_python` cases followed by 100 `multiple` cases; both are single-turn, single-call English AST categories. The returned `BfclBatch` retains unmodified upstream prompts and ground truths by upstream ID. `EvaluationExample.expected_arguments` is one representative answer because the internal schema cannot express BFCL's alternative values and optional arguments. **Use BFCL's [official AST evaluator](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py) with those original records and saved raw model outputs for BFCL accuracy**, not `sila.evaluation.score_examples`. The adapter changes BFCL's top-level parameter type spelling from `dict` to JSON Schema `object` only in `ToolDefinition`; its preserved prompt is unchanged.

Both loaders require an explicit immutable upstream revision and only read local files. Pin and obtain the data outside the unit tests. Unsupported BFCL categories: parallel calls, irrelevance/relevance, Java/JavaScript, multi-turn, web search, memory, format sensitivity, and other executable or agentic categories. The full BFCL package currently pins NumPy 1.26.4, which does not support this project's Python 3.14 environment; run its official evaluator in a separate supported environment rather than copying its scoring logic here.

## Initial success criteria

The treatment must satisfy all three conditions:

1. Improve Arabic exact tool-call accuracy by at least 5 percentage points **or** reduce relative error by at least 25%.
2. Regress by no more than 2 percentage points on English BFCL accuracy.
3. Show no reduction in Arabic negative-case accuracy.

## Limitations

The evaluation measures structural correctness, not real tool execution or downstream task success. Results from one 4B checkpoint and one QLoRA setup may not generalize to other model sizes or training methods. Coverage of Arabic dialects, domains, ambiguous requests, and adversarial inputs will be limited by the final evaluation sets. Quantization and hardware can also affect reproducibility despite fixed seeds and pinned revisions.
