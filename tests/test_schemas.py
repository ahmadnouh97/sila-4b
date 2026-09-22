from dataclasses import replace
from math import nan

import pytest

from sila.schemas import (
    EvaluationExample,
    EvaluationResult,
    ExpectedToolCall,
    ParsedToolCall,
    RawPrediction,
    ToolDefinition,
)


@pytest.fixture
def weather_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Return weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )


def test_valid_records_preserve_raw_prediction(weather_tool: ToolDefinition) -> None:
    example = EvaluationExample(
        id="ar-weather-001",
        source="handwritten-v1",
        language="ar",
        dialect="msa",
        domain="weather",
        user_utterance="ما الطقس في عمّان؟",
        available_tools=(weather_tool,),
        should_call_tool=True,
        expected_tool_name="get_weather",
        expected_arguments={"city": "عمّان"},
        tags=("unseen-tool",),
    )
    expected = ExpectedToolCall(example.expected_tool_name, example.expected_arguments)
    parsed = ParsedToolCall("get_weather", {"city": "عمّان"})
    raw = RawPrediction(
        example_id=example.id,
        generated_text='{"name":"get_weather","arguments":{"city":"عمّان"}}',
        model_id="org/model-4b",
        model_revision="0123456789abcdef",
        prompt_version="tool-prompt-v1",
        generation_config={"do_sample": False, "max_new_tokens": 128},
        latency_seconds=0.25,
        gpu_memory_mb=2048.0,
    )
    result = EvaluationResult(
        example_id=example.id,
        raw_prediction=raw,
        expected_call=expected,
        parsed_call=parsed,
        tool_selection_correct=True,
        arguments_correct=True,
        exact_match=True,
    )

    assert result.raw_prediction.generated_text == raw.generated_text


def test_valid_negative_example_and_empty_generation(
    weather_tool: ToolDefinition,
) -> None:
    example = EvaluationExample(
        id="ar-negative-001",
        source="handwritten-v1",
        language="ar",
        domain="weather",
        user_utterance="أحب الطقس المعتدل.",
        available_tools=(weather_tool,),
        should_call_tool=False,
        expected_tool_name=None,
        expected_arguments=None,
    )
    raw = RawPrediction(
        example_id=example.id,
        generated_text="",
        model_id="org/model-4b",
        model_revision="0123456789abcdef",
        prompt_version="tool-prompt-v1",
        generation_config={},
        latency_seconds=0,
    )

    assert raw.generated_text == ""


@pytest.mark.parametrize(
    ("should_call", "name", "arguments", "message"),
    [
        (True, None, None, "tool_name"),
        (True, "missing_tool", {}, "available_tools"),
        (False, "get_weather", {}, "negative examples"),
    ],
)
def test_rejects_inconsistent_examples(
    weather_tool: ToolDefinition,
    should_call: bool,
    name: str | None,
    arguments: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EvaluationExample(
            id="bad-example",
            source="test",
            language="ar",
            domain="weather",
            user_utterance="اختبار",
            available_tools=(weather_tool,),
            should_call_tool=should_call,
            expected_tool_name=name,
            expected_arguments=arguments,
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"type": "array", "properties": {}},
        {"type": "object", "properties": {}, "default": {1: "bad key"}},
    ],
)
def test_rejects_malformed_tool_schemas(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ToolDefinition("bad", "Bad schema", parameters)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_revision": ""}, "model_revision"),
        ({"generation_config": {"temperature": nan}}, "generation_config"),
        ({"latency_seconds": -0.1}, "latency_seconds"),
        ({"gpu_memory_mb": nan}, "gpu_memory_mb"),
    ],
)
def test_rejects_malformed_raw_predictions(
    changes: dict[str, object], message: str
) -> None:
    values = {
        "example_id": "example-1",
        "generated_text": "raw response",
        "model_id": "org/model-4b",
        "model_revision": "0123456789abcdef",
        "prompt_version": "v1",
        "generation_config": {},
        "latency_seconds": 0.1,
    }

    with pytest.raises(ValueError, match=message):
        RawPrediction(**(values | changes))


def test_rejects_inconsistent_results() -> None:
    raw = RawPrediction(
        example_id="example-1",
        generated_text="raw response",
        model_id="org/model-4b",
        model_revision="0123456789abcdef",
        prompt_version="v1",
        generation_config={},
        latency_seconds=0.1,
    )
    valid = EvaluationResult(
        example_id="example-1",
        raw_prediction=raw,
        expected_call=None,
        parsed_call=None,
        tool_selection_correct=True,
        arguments_correct=None,
        exact_match=True,
    )

    with pytest.raises(ValueError, match="IDs must match"):
        replace(valid, example_id="example-2")
    with pytest.raises(ValueError, match="exact_match"):
        replace(valid, tool_selection_correct=False)
