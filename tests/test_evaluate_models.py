import json
from pathlib import Path

import pytest

from sila.benchmark_metrics import SOURCES, score_arabfuncbench, verified_source
from sila.evaluate_models import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_VERSION,
    comparison,
    ensure_run_record,
    read_predictions,
    render_prompt,
)
from sila.evaluation import score_examples
from sila.schemas import EvaluationExample, RawPrediction, ToolDefinition


def test_prompt_uses_available_tools_without_gold_answers() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "ما الطقس؟"}]
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["tokenize"] is False
            assert kwargs["tools"] == [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            return "rendered"

    example = EvaluationExample(
        "case",
        "test",
        "ar",
        "weather",
        "ما الطقس؟",
        (ToolDefinition("weather", "Weather", {"type": "object", "properties": {}}),),
        True,
        "weather",
        {"secret_gold": "never render this"},
    )
    assert render_prompt(Tokenizer(), example) == "rendered"


def test_resume_refuses_changed_settings_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    ensure_run_record(path, {"max_new_tokens": 256})
    ensure_run_record(path, {"max_new_tokens": 256})
    with pytest.raises(ValueError, match="different"):
        ensure_run_record(path, {"max_new_tokens": 512})
    assert json.loads(path.read_text()) == {"max_new_tokens": 256}


def test_verdict_requires_all_three_criteria_and_complete_evaluation() -> None:
    base = {"exact_tool_call_accuracy": 0.8, "negative_case_accuracy": 0.9}
    adapter = {"exact_tool_call_accuracy": 0.85, "negative_case_accuracy": 0.9}
    result = comparison(base, adapter, 0.8, 0.78, complete=True)
    assert result["success"] is True
    assert result["arabic_relative_error_reduction"] == pytest.approx(0.25)
    assert comparison(base, adapter, 0.8, 0.77, complete=True)["success"] is False
    assert comparison(base, adapter, 0.8, 0.78, complete=False)["success"] is None
    adapter["negative_case_accuracy"] = 0.89
    assert comparison(base, adapter, 0.8, 0.78, complete=True)["success"] is False


def test_benchmark_bridge_preserves_duplicate_source_rows_and_invalid_outputs() -> None:
    tools = (
        ToolDefinition("weather", "Weather", {"type": "object", "properties": {}}),
    )
    examples = [
        EvaluationExample(
            f"repeated::row-{index}",
            "test",
            "ar",
            "services",
            f"prompt {index}",
            tools,
            False,
            None,
            None,
        )
        for index in range(2)
    ]
    raw = [
        RawPrediction(e.id, text, "model", "revision", "prompt", {}, 0)
        for e, text in zip(examples, ["plain refusal", '<tool_call>{"name":'])
    ]
    report = score_examples(examples, raw, output_format="qwen")
    rows = [
        {"id": "repeated", "utterance": "first"},
        {"id": "repeated", "utterance": "second"},
    ]
    # Isolate the format bridge: upstream metric files are ignored local inputs.
    namespace = {
        "evaluate_response": lambda row, response: {
            "utterance": row["utterance"],
            "tsa": int(response["function_called"] is None),
        },
        "compute_metrics": lambda results: {"correct": sum(r["tsa"] for r in results)},
    }
    result = score_arabfuncbench(rows, report, namespace)
    assert result["metrics"] == {"correct": 1}
    assert [r["utterance"] for r in result["results"]] == ["first", "second"]
    assert [r["prediction_id"] for r in result["results"]] == [
        "repeated::row-0",
        "repeated::row-1",
    ]


def test_modified_upstream_source_is_rejected(tmp_path: Path) -> None:
    name = "bfcl_ast_checker.py"
    (tmp_path / name).write_text("raise RuntimeError('must not execute')")
    assert name in SOURCES
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_source(tmp_path, name)


def test_resume_rejects_predictions_from_the_other_model(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    tools = (
        ToolDefinition("weather", "Weather", {"type": "object", "properties": {}}),
    )
    example = EvaluationExample(
        "case", "test", "ar", "weather", "prompt", tools, False, None, None
    )
    row = {
        "example_id": "case",
        "generated_text": "no call",
        "model_id": MODEL_ID + "+arabic-qlora",
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "generation_config": {"do_sample": False},
        "latency_seconds": 0,
        "gpu_memory_mb": None,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(read_predictions(path, (example,), "adapter", {"do_sample": False})) == 1
    with pytest.raises(ValueError, match="metadata differs"):
        read_predictions(path, (example,), "base", {"do_sample": False})
    assert json.loads(path.read_text()) == row
