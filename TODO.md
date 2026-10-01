# Next Qwen experiment

Goal: improve Arabic exact calls while preserving English and no-call accuracy.
The first adapter failed; results and completed work are in [README.md](README.md).
Preparation is complete; the owner has now authorized training in a separate chat.
Preparation is complete; the revised frozen-base development scores are recorded.
Improvement must be measured.

## Prepare

- [x] Freeze a small, reviewed Arabic/English development set and score the
  pinned NF4 base on it. Use independent domains, schemas, wording, and values;
  keep existing evaluation data out of training and checkpoint selection.
  Include normalization, enum grounding, fully specified requests, missing
  inputs, out-of-scope requests, and quoted/injected instructions. Record
  counts per category; a few smoke cases are not an accuracy evaluation.
- [x] When generation is requested, create a new versioned first-party Arabic
  corpus in a separate directory. Prioritize enum grounding, digits/dates,
  percentages/currency codes, and no-call decisions. Include contrasting
  examples where one changed input changes the argument or call decision.
  Increase meaningful no-call coverage from the current 9%; keep positive
  requests so the model also learns when clarification is unnecessary.
- [x] Review wording and targets, validate schema agreement, and separate
  training/validation/development by tool family and prompt template. Reject
  ambiguous examples; record human versus AI review honestly. Run leakage
  checks against development and all existing evaluations, including a review
  for semantic near-duplicates that exact/normalized checks miss.
- [x] Give the notebook, leakage audit, and paired runner explicit paths for
  the new manifest, data, audit, and outputs. Remove the notebook's v1.7-only
  restriction for this new version while retaining hash/review checks. Preserve
  the original manifest, reviewed test, adapter, predictions, and reports.
- [x] Check training labels on representative call/no-call examples: mask
  user/schema tokens, supervise arguments and the end-of-turn token, and match
  native inference prompts. Check rendered lengths across the new corpus so
  no examples are truncated. Reuse installed TRL and the existing scorer.
- [x] Support scoring saved checkpoints on the development set. Select by
  generated Arabic exact-call accuracy among checkpoints with no English or
  Arabic no-call regression versus the development base. Loss is diagnostic;
  if none qualifies, retain the base.

Prepared inputs: `data/followup-v1.8-dev2/` (9,060 train / 2,328 validation / 276
development rows, 29.14% no-call training). Review/leakage/tokenizer records:
`reports/followup-v1.8-dev2/`. Review is AI/template review, not native certification.
Base outputs: `outputs/development-v1.8/base-dev2/`; training remains disabled.
Development base: Arabic calls **57/66**, English calls **62/66**, Arabic no-call
**54/72**. Advance at **60/66**, **62/66**, and **54/72** respectively. The initial
252-case set had perfect calls; a fixed reviewed 24-case challenge appendix
was frozen before its inference. Both versions are preserved.

## Run in the authorized training chat

- [ ] Train one fresh adapter from the pinned Qwen base on the reviewed new
  corpus: trial learning rate `5e-5`, at most 330 optimizer updates, checkpoints
  at 110/220/330. Keep NF4, LoRA rank 16/alpha 32/dropout 0.05, target modules,
  effective batch 16, and seed 42; record the complete configuration. Lower
  learning rate and a shorter budget are hypotheses, not proven fixes. This
  run tests the combined recipe; it cannot isolate which change caused a gain.
- [ ] Advance only if development Arabic calls gain at least 5 percentage
  points or errors fall at least 25%, without English/no-call regression.
  Freeze the candidate before running the unchanged final evaluations.
- [ ] Require the preset final gates: Arabic calls **at least 61/70**, BFCL
  **at least 459/500**, and Arabic no-call **at least 17/30**. Aim for **62/70,
  469/500, and above 17/30**. Preserve complete raw responses and paired prompt/
  settings hashes; use official BFCL AST and pinned external metrics. Report
  regressions and failures. The inspected Arabic test is a historical
  comparison, not an unseen test; broader claims need fresh confirmation.

## Separate follow-up

- [ ] Recover the original missing ArabFuncBench base response
  `education_neg_4_008`, or retain the documented incomplete historical result.
  Never silently replace it. This does not block the next training experiment.
