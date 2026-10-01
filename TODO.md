# Experiment status

- [x] Complete QLoRA training and save the adapter and run record.
- [x] Build paired evaluation and preserve generated responses.
- [x] Score the reviewed Arabic set, ArabFuncBench, and fixed BFCL subset.
- [x] Record the failed acceptance criteria and failure analysis.
- [x] Update the README and reproduction guide.

## Follow-up work

- [ ] Investigate the missing ArabFuncBench base raw response. Its original metric record remains available; do not claim complete external reproducibility or silently replace it.
- [ ] Before considering another training run, review synthetic data quality and normalization/no-call coverage using independent development cases.
- [ ] If pursuing JAIS, review and approve an immutable revision, license, and custom code before download or use.

No follow-up training is scheduled. Preserve the current manifest, reviewed evaluation set, and run artifacts.
