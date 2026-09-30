"""Deterministic scoring for JSON and native Qwen tool-call responses.

Tool and argument rates use positive examples; negative rates use negative
examples. Argument key metrics compare top-level keys on positive examples.
Undefined rates (an empty denominator) are reported as 0.0.
"""

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from sila.schemas import (
    EvaluationExample,
    EvaluationResult,
    ExpectedToolCall,
    JsonValue,
    ParsedToolCall,
    RawPrediction,
)


def _no_duplicates(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _response_value(text: str, output_format: str) -> tuple[JsonValue, bool]:
    """Parse a response, allowing Qwen call blocks or plain no-call text."""
    text = text.strip()
    wrapped = False
    if output_format == "qwen" and ("<tool_call" in text or "</tool_call" in text):
        if (
            text.count("<tool_call") != 1
            or text.count("</tool_call") != 1
            or "<tool_call>" not in text
            or "</tool_call>" not in text
        ):
            raise ValueError("expected one complete Qwen tool-call block")
        before, _, rest = text.partition("<tool_call>")
        text, _, after = rest.partition("</tool_call>")
        if "</tool_call" in before or "<tool_call" in after:
            raise ValueError("invalid Qwen tool-call marker order")
        wrapped = True
    try:
        value = json.loads(
            text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant
        )
    except json.JSONDecodeError:
        if output_format == "qwen" and not wrapped and text and text[0] not in '{["`<':
            # ponytail: plain prose means no call; reply quality needs manual review.
            return None, False
        raise
    if wrapped and not isinstance(value, dict):
        raise ValueError("Qwen tool-call block must contain a call object")
    return value, True


def _equal(left: JsonValue, right: JsonValue) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return Decimal(str(left)) == Decimal(str(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


@dataclass(frozen=True, slots=True)
class ScoredExample:
    result: EvaluationResult
    parse_success: bool
    valid_structured_output: bool
    should_call_correct: bool
    argument_key_tp: int
    argument_key_fp: int
    argument_key_fn: int


@dataclass(frozen=True, slots=True)
class ScoreReport:
    examples: tuple[ScoredExample, ...]
    counts: dict[str, int]
    rates: dict[str, float]


def score_examples(
    examples: Sequence[EvaluationExample],
    predictions: Sequence[RawPrediction],
    *,
    output_format: str = "json",
) -> ScoreReport:
    """Score one raw prediction per example without executing generated calls.

    ``name`` and ``arguments`` are the only call fields. JSON ``null`` is the
    refusal. ``output_format='qwen'`` also accepts one native ``<tool_call>``
    block or non-empty plain prose as a no-call decision. Plain prose does not
    count as parsed JSON or valid structured output; its helpfulness is not
    scored. Malformed or multiple call blocks are invalid, never refusals.
    Select the same format for base and adapter, independently of gold labels.
    """
    if output_format not in {"json", "qwen"}:
        raise ValueError("output_format must be 'json' or 'qwen'")
    expected_by_id = {example.id: example for example in examples}
    predicted_by_id = {prediction.example_id: prediction for prediction in predictions}
    if len(expected_by_id) != len(examples) or len(predicted_by_id) != len(predictions):
        raise ValueError("duplicate example or prediction ID")
    if expected_by_id.keys() != predicted_by_id.keys():
        raise ValueError("example and prediction IDs must match")

    scored = []
    for example in examples:
        raw = predicted_by_id[example.id]
        parsed_call = None
        parse_success = False
        valid = False
        parse_error = None
        try:
            value, parse_success = _response_value(raw.generated_text, output_format)
            if value is None:
                valid = True
            elif (
                isinstance(value, dict)
                and value.keys() == {"name", "arguments"}
                and isinstance(value["name"], str)
                and value["name"].strip()
                and isinstance(value["arguments"], dict)
            ):
                parsed_call = ParsedToolCall(value["name"], value["arguments"])
                valid = True
            else:
                parse_error = "invalid structured output"
        except (json.JSONDecodeError, ValueError) as error:
            parse_error = str(error)

        expected_call = (
            ExpectedToolCall(example.expected_tool_name, example.expected_arguments)
            if example.should_call_tool
            else None
        )
        should_call_correct = valid and (
            (parsed_call is not None) == example.should_call_tool
        )
        tool_correct = (
            parsed_call.tool_name == expected_call.tool_name
            if parsed_call is not None and expected_call is not None
            else should_call_correct and expected_call is None
        )
        arguments_correct = (
            (
                _equal(parsed_call.arguments, expected_call.arguments)
                if parsed_call is not None and expected_call is not None
                else False
            )
            if expected_call is not None
            else None
        )
        actual_keys = set(parsed_call.arguments) if parsed_call else set()
        expected_keys = set(expected_call.arguments) if expected_call else set()
        result = EvaluationResult(
            example_id=example.id,
            raw_prediction=raw,
            expected_call=expected_call,
            parsed_call=parsed_call,
            tool_selection_correct=tool_correct,
            arguments_correct=arguments_correct,
            exact_match=tool_correct and (arguments_correct is not False),
            parse_error=parse_error,
        )
        scored.append(
            ScoredExample(
                result=result,
                parse_success=parse_success,
                valid_structured_output=valid and parse_success,
                should_call_correct=should_call_correct,
                argument_key_tp=len(actual_keys & expected_keys),
                argument_key_fp=len(actual_keys - expected_keys)
                if expected_call
                else 0,
                argument_key_fn=len(expected_keys - actual_keys),
            )
        )

    positive = [item for item in scored if item.result.expected_call is not None]
    negative = [item for item in scored if item.result.expected_call is None]
    counts = {
        "examples": len(scored),
        "positive_examples": len(positive),
        "negative_examples": len(negative),
        "parse_success": sum(item.parse_success for item in scored),
        "valid_structured_output": sum(item.valid_structured_output for item in scored),
        "should_call_correct": sum(item.should_call_correct for item in scored),
        "tool_name_correct": sum(
            item.result.tool_selection_correct for item in positive
        ),
        "exact_arguments_correct": sum(
            item.result.arguments_correct is True for item in positive
        ),
        "exact_tool_calls_correct": sum(item.result.exact_match for item in positive),
        "exact_match": sum(item.result.exact_match for item in scored),
        "negative_case_correct": sum(item.result.exact_match for item in negative),
        "false_calls": sum(item.result.parsed_call is not None for item in negative),
        "argument_key_tp": sum(item.argument_key_tp for item in positive),
        "argument_key_fp": sum(item.argument_key_fp for item in positive),
        "argument_key_fn": sum(item.argument_key_fn for item in positive),
    }

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    tp = counts["argument_key_tp"]
    fp = counts["argument_key_fp"]
    fn = counts["argument_key_fn"]
    rates = {
        "parse_success": rate(counts["parse_success"], counts["examples"]),
        "valid_structured_output": rate(
            counts["valid_structured_output"], counts["examples"]
        ),
        "should_call_accuracy": rate(counts["should_call_correct"], counts["examples"]),
        "tool_name_accuracy": rate(
            counts["tool_name_correct"], counts["positive_examples"]
        ),
        "exact_argument_accuracy": rate(
            counts["exact_arguments_correct"], counts["positive_examples"]
        ),
        "exact_tool_call_accuracy": rate(
            counts["exact_tool_calls_correct"], counts["positive_examples"]
        ),
        "argument_key_precision": rate(tp, tp + fp),
        "argument_key_recall": rate(tp, tp + fn),
        "argument_key_f1": rate(2 * tp, 2 * tp + fp + fn),
        "negative_case_accuracy": rate(
            counts["negative_case_correct"], counts["negative_examples"]
        ),
        "false_call_rate": rate(counts["false_calls"], counts["negative_examples"]),
    }
    return ScoreReport(tuple(scored), counts, rates)
