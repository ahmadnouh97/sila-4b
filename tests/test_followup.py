import json

import pytest

from sila.audit_leakage import run_audit
from sila.development import select_checkpoint, verified_checkpoint
from sila.evaluate_models import sha256
from sila.followup_data import build_splits, to_example
from sila.validate_stress_set import _check_schema, _check_value


def test_followup_is_deterministic_separate_and_schema_grounded() -> None:
    splits = build_splits()
    assert splits == build_splits()
    names = {}
    prompts = set()
    for split, rows in splits.items():
        names[split] = set()
        for row in rows:
            example = to_example(row)
            assert example.user_utterance not in prompts
            prompts.add(example.user_utterance)
            if split != "development":
                assert example.language == "ar"
            for tool in example.available_tools:
                names[split].add(tool.name)
                _check_schema(tool.parameters)
            if example.should_call_tool:
                tool = next(
                    t
                    for t in example.available_tools
                    if t.name == example.expected_tool_name
                )
                _check_value(example.expected_arguments, tool.parameters, "target")
                assert row["messages"][-1]["tool_calls"][0]["function"] == {
                    "name": tool.name,
                    "arguments": example.expected_arguments,
                }
            else:
                assert row["messages"][-1]["content"]
    assert not names["train"] & names["validation"]
    assert not names["train"] & names["development"]
    assert not names["validation"] & names["development"]
    assert {r["language"] for r in splits["development"]} == {"ar", "en"}
    assert sum(not r["should_call_tool"] for r in splits["train"]) > (
        len(splits["train"]) * 0.2
    )
    row = dict(splits["development"][0], expected_arguments={"bad": True})
    with pytest.raises(ValueError):
        to_example(row)
    modes = {tag for r in splits["development"] for tag in r["tags"]}
    assert {"outside", "unsupported", "contrast"} <= modes
    for row in splits["development"]:
        if row["tags"] == ["contrast"]:
            direct = next(
                r
                for r in splits["development"]
                if r["id"] == row["id"].replace("-contrast", "-direct")
            )
            assert (
                sum(
                    row["expected_arguments"][k] != v
                    for k, v in direct["expected_arguments"].items()
                )
                == 1
            )
    row = dict(splits["development"][0])
    row["messages"] = [row["messages"][0], {"role": "assistant", "content": "wrong"}]
    with pytest.raises(ValueError, match="target"):
        to_example(row)


def test_new_corpus_cannot_overwrite_historical_audit(tmp_path) -> None:
    with pytest.raises(ValueError, match="separate audit"):
        run_audit(data_dir=tmp_path)


def test_selection_requires_improvement_and_both_preservation_gates() -> None:
    base = {
        "ar": {"exact_tool_call_accuracy": 0.8, "negative_case_accuracy": 0.8},
        "en": {"exact_tool_call_accuracy": 0.9},
    }
    candidate = {
        "ar": {"exact_tool_call_accuracy": 0.85, "negative_case_accuracy": 0.8},
        "en": {"exact_tool_call_accuracy": 0.9},
    }
    assert select_checkpoint(base, {"checkpoint-110": candidate}) == "checkpoint-110"
    candidate["en"]["exact_tool_call_accuracy"] = 0.89
    assert select_checkpoint(base, {"checkpoint-110": candidate}) is None

    candidate["en"]["exact_tool_call_accuracy"] = 0.9
    candidate["ar"]["negative_case_accuracy"] = 0.79
    assert select_checkpoint(base, {"checkpoint-110": candidate}) is None
    candidate["ar"]["negative_case_accuracy"] = 0.8
    candidate["ar"]["exact_tool_call_accuracy"] = 0.81
    assert select_checkpoint(base, {"checkpoint-110": candidate}) is None


def test_selection_rejects_weights_changed_since_scoring(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    weights.write_bytes(b"original")
    config.write_text("{}")
    record = {
        "adapter": str(adapter),
        "adapter_sha256": sha256(weights),
        "adapter_config_sha256": sha256(config),
    }
    (tmp_path / "run.json").write_text(json.dumps(record))
    assert verified_checkpoint(tmp_path) == record
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verified_checkpoint(tmp_path)
