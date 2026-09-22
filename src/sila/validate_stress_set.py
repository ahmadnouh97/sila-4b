"""Validate the stress-set template, pilot, 100-case draft, or reviewed set."""

import argparse
import json
import re
from collections import Counter
from datetime import date, time
from pathlib import Path

from sila.schemas import EvaluationExample, ToolDefinition

CATEGORIES = (
    "direct_request",
    "dialect_request",
    "code_switching",
    "orthographic_noise",
    "distractor_tools",
    "multi_argument",
    "argument_normalization",
    "missing_required_input",
    "out_of_scope",
    "prompt_injection",
)
NEGATIVE_CATEGORIES = {"missing_required_input", "out_of_scope", "prompt_injection"}
DIALECTS = {"msa", "egyptian", "levantine", "gulf", "iraqi", "maghrebi"}
SCALAR_TYPES = {"string", "integer", "number", "boolean", "null"}
# ponytail: covers current date words; expand for new dialect phrases before freezing.
RELATIVE_DATE_PATTERN = re.compile(r"(?<!\w)(?:بكرة|بكرا|بكره|غداً|غدا|باجر|باچر)(?!\w)")


def _check_schema(schema: object) -> None:
    if not isinstance(schema, dict):
        raise ValueError("tool schema must be an object")
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict) or any(
            not isinstance(key, str) or not key.strip() for key in properties
        ):
            raise ValueError("object schema needs named properties")
        for child in properties.values():
            _check_schema(child)
        required = schema.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(key, str) for key in required)
            or len(required) != len(set(required))
            or not set(required) <= properties.keys()
        ):
            raise ValueError("required keys must be unique declared properties")
        if schema.get("additionalProperties") is not False:
            raise ValueError("object schema must forbid additionalProperties")
    elif kind == "array":
        _check_schema(schema.get("items"))
    elif kind not in SCALAR_TYPES:
        raise ValueError(f"unsupported tool schema type: {kind}")
    if "description" in schema and not isinstance(schema["description"], str):
        raise ValueError("schema description must be a string")
    if "enum" in schema and (
        not isinstance(schema["enum"], list) or not schema["enum"]
    ):
        raise ValueError("schema enum must be a non-empty list")
    if "format" in schema and (
        kind != "string" or schema["format"] not in {"date", "time"}
    ):
        raise ValueError("unsupported schema format")
    if "minimum" in schema and (
        kind not in {"integer", "number"}
        or isinstance(schema["minimum"], bool)
        or not isinstance(schema["minimum"], (int, float))
    ):
        raise ValueError("invalid schema minimum")


def _check_value(value: object, schema: dict, field: str) -> None:
    kind = schema["type"]
    valid = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float))
        and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
        "array": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
    }[kind]()
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise ValueError(f"{field} violates tool schema")
    if "minimum" in schema and value < schema["minimum"]:
        raise ValueError(f"{field} is below schema minimum")
    if schema.get("format") == "date":
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError(f"{field} must use YYYY-MM-DD")
    elif schema.get("format") == "time":
        if time.fromisoformat(value).strftime("%H:%M") != value:
            raise ValueError(f"{field} must use HH:MM")
    if kind == "array":
        for item in value:
            _check_value(item, schema["items"], field)
    elif kind == "object":
        if not value.keys() <= schema["properties"].keys():
            raise ValueError(f"{field} has undeclared keys")
        if not set(schema.get("required", [])) <= value.keys():
            raise ValueError(f"{field} omits required keys")
        for key, item in value.items():
            _check_value(item, schema["properties"][key], f"{field}.{key}")


def validate_stress_set(
    path: Path, *, final: bool = False, pilot: bool = False, draft: bool = False
) -> dict[str, int]:
    """Raise ValueError unless every row and category count matches its mode."""
    if sum((final, pilot, draft)) > 1:
        raise ValueError("choose one validation mode")
    counts: Counter[str] = Counter()
    ids: set[str] = set()
    utterances: set[str] = set()
    reference_date = None
    known_tools = {}
    with path.open(encoding="utf-8", errors="strict") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                category = row["category"]
                if category not in CATEGORIES:
                    raise ValueError(f"unknown category: {category}")
                if row["dialect"] not in DIALECTS:
                    raise ValueError(f"unknown dialect: {row['dialect']}")
                expected_status = (
                    "native_speaker_reviewed"
                    if final
                    else "draft_not_gold"
                    if draft
                    else "pilot_draft_not_gold"
                    if pilot
                    else "illustrative_not_gold"
                )
                if row["review_status"] != expected_status:
                    raise ValueError(f"review_status must be {expected_status}")
                if row.get("language", "ar") != "ar" or (
                    (pilot or draft or final) and "language" not in row
                ):
                    raise ValueError("language must be ar")
                if pilot or draft or final:
                    current_date = row["reference_date"]
                    date.fromisoformat(current_date)
                    if reference_date is None:
                        reference_date = current_date
                    elif current_date != reference_date:
                        raise ValueError("reference_date must be fixed across set")
                if final and "review_note" in row:
                    raise ValueError("final cases cannot retain review_note")
                if "review_note" in row and (
                    not isinstance(row["review_note"], str)
                    or not row["review_note"].strip()
                ):
                    raise ValueError("review_note must be non-empty text")
                utterance = row["user_utterance"]
                if not isinstance(utterance, str) or not any(
                    "\u0600" <= char <= "\u06ff"
                    or "\u0750" <= char <= "\u077f"
                    or "\u08a0" <= char <= "\u08ff"
                    for char in utterance
                ):
                    raise ValueError("user_utterance must contain Arabic text")
                if utterance in utterances:
                    raise ValueError("duplicate user_utterance")
                if (
                    (pilot or draft or final)
                    and RELATIVE_DATE_PATTERN.search(utterance)
                    and current_date not in utterance
                ):
                    raise ValueError(
                        "relative-date utterance must include reference_date"
                    )
                tools = tuple(ToolDefinition(**tool) for tool in row["available_tools"])
                for tool in tools:
                    _check_schema(tool.parameters)
                    if tool.name in known_tools and known_tools[tool.name] != tool:
                        raise ValueError(f"tool definition changed: {tool.name}")
                    known_tools[tool.name] = tool
                example = EvaluationExample(
                    id=row["id"],
                    source="manual_arabic_stress_set",
                    language="ar",
                    domain=category,
                    user_utterance=utterance,
                    available_tools=tools,
                    should_call_tool=row["should_call_tool"],
                    expected_tool_name=row["expected_tool_name"],
                    expected_arguments=row["expected_arguments"],
                    dialect=row["dialect"],
                )
                if example.id in ids:
                    raise ValueError(f"duplicate ID: {example.id}")
                if example.should_call_tool == (category in NEGATIVE_CATEGORIES):
                    raise ValueError(
                        f"inconsistent positive/negative category: {category}"
                    )
                if example.should_call_tool:
                    if (pilot or draft or final) and len(tools) < 2:
                        raise ValueError(
                            "positive case needs competing available tools"
                        )
                    selected = next(
                        tool
                        for tool in tools
                        if tool.name == example.expected_tool_name
                    )
                    properties = selected.parameters["properties"]
                    if not example.expected_arguments.keys() <= properties.keys():
                        raise ValueError(
                            "expected argument keys are not permitted by tool"
                        )
                    if (
                        not set(selected.parameters.get("required", []))
                        <= example.expected_arguments.keys()
                    ):
                        raise ValueError("expected arguments omit required tool keys")
                    _check_value(
                        example.expected_arguments,
                        selected.parameters,
                        "expected_arguments",
                    )
                ids.add(example.id)
                utterances.add(utterance)
                counts[category] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"line {line_number}: {error}") from error
    per_category = 10 if final or draft else 3 if pilot else 1
    wrong = {
        category: counts[category]
        for category in CATEGORIES
        if counts[category] != per_category
    }
    if wrong:
        raise ValueError(f"expected {per_category} per category, got {wrong}")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="require 30 draft cases")
    mode.add_argument("--draft", action="store_true", help="require 100 draft cases")
    mode.add_argument("--final", action="store_true", help="require 100 reviewed cases")
    args = parser.parse_args()
    try:
        counts = validate_stress_set(
            args.path, final=args.final, pilot=args.pilot, draft=args.draft
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(f"Valid stress set: {args.path} ({sum(counts.values())} cases; {counts})")


if __name__ == "__main__":
    main()
