import json
from dataclasses import replace

import pytest

from sila.evaluation import score_examples
from sila.schemas import EvaluationExample, RawPrediction, ToolDefinition

TOOLS = (
    ToolDefinition("weather", "Weather", {"type": "object", "properties": {}}),
    ToolDefinition("search", "Search", {"type": "object", "properties": {}}),
)


def example(id: str, arguments: dict | None) -> EvaluationExample:
    return EvaluationExample(
        id=id,
        source="test",
        language="ar",
        domain="weather",
        user_utterance="ما الطقس؟",
        available_tools=TOOLS,
        should_call_tool=arguments is not None,
        expected_tool_name="weather" if arguments is not None else None,
        expected_arguments=arguments,
    )


def prediction(id: str, text: str) -> RawPrediction:
    return RawPrediction(id, text, "model", "revision", "prompt", {}, 0.0)


def call(arguments: dict, name: str = "weather") -> str:
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def test_hand_calculated_fixture_metrics() -> None:
    # 12 examples: 9 positive, 3 negative. Key totals: TP=9, FP=1, FN=3.
    cases = [
        ("correct", {"city": "عمان", "days": 3}, call({"city": "عمان", "days": 3})),
        ("wrong_tool", {"city": "عمان"}, call({"city": "عمان"}, "search")),
        ("missing", {"city": "عمان", "days": 3}, call({"city": "عمان"})),
        ("extra", {"city": "عمان"}, call({"city": "عمان", "unit": "C"})),
        ("wrong_value", {"city": "عمان"}, call({"city": "دمشق"})),
        ("malformed", {"city": "عمان"}, '{"name":"weather",'),
        ("correct_refusal", None, "null"),
        ("incorrect_refusal", {"city": "عمان"}, "null"),
        (
            "key_order",
            {"city": "عمان", "options": {"unit": "C", "days": 3}},
            '{"arguments":{"options":{"days":3,"unit":"C"},"city":"عمان"},"name":"weather"}',
        ),
        ("arabic", {"city": "عَمَّان"}, call({"city": "عَمَّان"})),
        ("false_call", None, call({"city": "عمان"})),
        ("invalid_shape", None, '"hello"'),
    ]
    report = score_examples(
        [example(id, args) for id, args, _ in cases],
        [prediction(id, text) for id, _, text in cases],
    )

    assert report.counts == {
        "examples": 12,
        "positive_examples": 9,
        "negative_examples": 3,
        "parse_success": 11,
        "valid_structured_output": 10,
        "should_call_correct": 8,
        "tool_name_correct": 6,
        "exact_arguments_correct": 4,
        "exact_tool_calls_correct": 3,
        "exact_match": 4,
        "negative_case_correct": 1,
        "false_calls": 1,
        "argument_key_tp": 9,
        "argument_key_fp": 1,
        "argument_key_fn": 3,
    }
    assert report.rates == pytest.approx(
        {
            "parse_success": 11 / 12,
            "valid_structured_output": 10 / 12,
            "should_call_accuracy": 8 / 12,
            "tool_name_accuracy": 6 / 9,
            "exact_argument_accuracy": 4 / 9,
            "exact_tool_call_accuracy": 3 / 9,
            "argument_key_precision": 9 / 10,
            "argument_key_recall": 9 / 12,
            "argument_key_f1": 18 / 22,
            "negative_case_accuracy": 1 / 3,
            "false_call_rate": 1 / 3,
        }
    )
    by_id = {item.result.example_id: item for item in report.examples}
    assert {id for id, item in by_id.items() if item.result.exact_match} == {
        "correct",
        "correct_refusal",
        "key_order",
        "arabic",
    }
    assert by_id["missing"].argument_key_fn == 1
    assert by_id["extra"].argument_key_fp == 1
    assert by_id["wrong_value"].result.arguments_correct is False
    assert by_id["malformed"].result.parse_error
    assert not by_id["malformed"].parse_success
    assert (
        by_id["malformed"].result.raw_prediction.generated_text == '{"name":"weather",'
    )
    assert by_id["invalid_shape"].parse_success
    assert not by_id["invalid_shape"].valid_structured_output
    assert not by_id["invalid_shape"].should_call_correct
    assert by_id["wrong_tool"].result.arguments_correct is True
    assert not by_id["wrong_tool"].result.exact_match
    assert by_id["correct_refusal"].result.exact_match
    assert by_id["incorrect_refusal"].result.arguments_correct is False


def test_only_equivalent_primitive_values_match() -> None:
    expected = example("number", {"days": 3, "active": True, "city": "عَمَّان"})
    matching = score_examples(
        [expected],
        [prediction("number", call({"days": 3.0, "active": True, "city": "عَمَّان"}))],
    )
    assert matching.examples[0].result.arguments_correct is True

    for arguments in (
        {"days": True, "active": True, "city": "عَمَّان"},
        {"days": 3, "active": 1, "city": "عَمَّان"},
        {"days": 3, "active": True, "city": "عمان"},
        {"days": 3, "active": True, "city": " عَمَّان "},
    ):
        report = score_examples([expected], [prediction("number", call(arguments))])
        assert report.examples[0].result.arguments_correct is False


def test_rejects_missing_duplicate_and_extra_predictions() -> None:
    expected = example("one", {})
    raw = prediction("one", call({}))
    for examples, predictions in (
        ([expected], []),
        ([expected], [raw, raw]),
        ([expected, expected], [raw]),
        ([expected], [replace(raw, example_id="two")]),
    ):
        with pytest.raises(ValueError):
            score_examples(examples, predictions)


def test_duplicate_json_keys_do_not_silently_overwrite() -> None:
    report = score_examples(
        [example("one", {"city": "عمان"})],
        [
            prediction(
                "one", '{"name":"weather","arguments":{"city":"عمان","city":"دمشق"}}'
            )
        ],
    )
    assert report.counts["parse_success"] == 0
    assert report.examples[0].result.parse_error == "duplicate JSON key: city"


def test_qwen_native_response_contract() -> None:
    wrapped = f"<tool_call>\n{call({'city': 'عمان'})}\n</tool_call>"
    cases = [
        ({"city": "عمان"}, wrapped, True, True),
        ({"city": "عمان"}, "سأبحث الآن.\n" + wrapped, True, True),
        (None, "ما المدينة التي تقصدها؟", True, False),
        (None, "null", True, True),
        ({"city": "عمان"}, "ما المدينة التي تقصدها؟", False, False),
        (None, wrapped, False, True),
        (None, "", False, False),
        (None, "<tool_call>{broken}</tool_call>", False, False),
        (None, "<tool_call>" + call({}), False, False),
        (None, "</tool_call>", False, False),
        (None, "</tool_call><tool_call>{}", False, False),
        (None, "<tool_call>null</tool_call>", False, False),
        (None, "<tool_call>[]</tool_call>", False, False),
        (None, "<tool_call>{}</tool_call>", False, False),
        (None, '<tool_call type="function">{}</tool_call>', False, False),
        ({"city": "عمان"}, wrapped + wrapped, False, False),
        (None, '{"name":"weather",', False, False),
        (None, '"hello"', False, False),
        (None, "```json\n{}\n```", False, False),
        (
            None,
            '<tool_call>{"name":"weather","arguments":{},"arguments":{}}</tool_call>',
            False,
            False,
        ),
    ]
    for args, text, exact, structured in cases:
        raw = prediction("one", text)
        item = score_examples(
            [example("one", args)], [raw], output_format="qwen"
        ).examples[0]
        assert item.result.exact_match is exact, text
        assert item.valid_structured_output is structured, text
        assert item.result.raw_prediction == raw
        if text == "ما المدينة التي تقصدها؟":
            assert not item.parse_success
            assert item.result.parse_error is None
        if not exact and not structured and args is None:
            assert item.result.parse_error

    with pytest.raises(ValueError, match="output_format"):
        score_examples([], [], output_format="unknown")
    strict = score_examples(
        [example("one", None)], [prediction("one", "ما المدينة التي تقصدها؟")]
    ).examples[0]
    assert not strict.result.exact_match
