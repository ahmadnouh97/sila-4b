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

## Initial success criteria

The treatment must satisfy all three conditions:

1. Improve Arabic exact tool-call accuracy by at least 5 percentage points **or** reduce relative error by at least 25%.
2. Regress by no more than 2 percentage points on English BFCL accuracy.
3. Show no reduction in Arabic negative-case accuracy.

## Limitations

The evaluation measures structural correctness, not real tool execution or downstream task success. Results from one 4B checkpoint and one QLoRA setup may not generalize to other model sizes or training methods. Coverage of Arabic dialects, domains, ambiguous requests, and adversarial inputs will be limited by the final evaluation sets. Quantization and hardware can also affect reproducibility despite fixed seeds and pinned revisions.
