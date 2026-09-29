"""Read-only adapters for locally supplied, revision-pinned benchmark files.

BFCL's possible answers contain alternatives and optional arguments. The
``EvaluationExample`` call is only a representative for the internal prompt
contract; use ``BfclBatch.ground_truth`` with BFCL's official AST evaluator
for BFCL accuracy. These adapters never execute benchmark code or tool calls.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from sila.schemas import EvaluationExample, ToolDefinition

BFCL_CATEGORIES = ("simple_python", "multiple")


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_arabfuncbench(
    examples_path: Path, tools_path: Path, *, revision: str
) -> tuple[EvaluationExample, ...]:
    """Load ArabFuncBench's gated JSON exports for evaluation only."""
    if not revision.strip():
        raise ValueError("revision must be non-empty")
    with tools_path.open(encoding="utf-8") as file:
        definitions = json.load(file)
    with examples_path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if isinstance(definitions, dict):
        if any(not isinstance(group, list) for group in definitions.values()):
            raise ValueError("ArabFuncBench tool groups must be lists")
        definitions = [tool for group in definitions.values() for tool in group]
    if not isinstance(definitions, list) or any(
        not isinstance(tool, dict) for tool in definitions
    ):
        raise ValueError("ArabFuncBench tools must be a list or grouped object")
    tools = {item["name"]: ToolDefinition(**item) for item in definitions}
    if len(tools) != len(definitions):
        raise ValueError("duplicate ArabFuncBench tool name")
    result = []
    for row in rows:
        negative = row["is_negative"]
        if not isinstance(negative, bool):
            raise ValueError(f"invalid is_negative for {row['id']}")
        result.append(
            EvaluationExample(
                id=row["id"],
                source=f"lsadouk1111/ArabFuncBench@{revision}",
                language="ar",
                dialect="msa",
                domain=row["domain"],
                user_utterance=row["utterance"],
                available_tools=tuple(tools[name] for name in row["available_tools"]),
                should_call_tool=not negative,
                expected_tool_name=row["expected_function"],
                expected_arguments=row["expected_arguments"],
                tags=("evaluation-only",),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class BfclBatch:
    examples: tuple[EvaluationExample, ...]
    ground_truth: dict[str, dict]
    prompts: dict[str, dict]


def load_bfcl(
    data_dir: Path,
    *,
    revision: str,
    limit: int = 500,
    categories: tuple[str, ...] = BFCL_CATEGORIES,
) -> BfclBatch:
    """Load a deterministic prefix of BFCL V4 single-call Python AST cases."""
    if not revision.strip():
        raise ValueError("revision must be non-empty")
    if limit < 1:
        raise ValueError("limit must be positive")
    if not categories or any(
        category not in BFCL_CATEGORIES for category in categories
    ):
        raise ValueError(f"supported BFCL categories: {BFCL_CATEGORIES}")
    examples = []
    ground_truth = {}
    prompts = {}
    for category in categories:
        if len(examples) == limit:
            break
        name = f"BFCL_v4_{category}.json"
        answer_rows = _jsonl(data_dir / "possible_answer" / name)
        answers = {item["id"]: item for item in answer_rows}
        if len(answers) != len(answer_rows):
            raise ValueError(f"duplicate BFCL answer ID in {name}")
        for row in _jsonl(data_dir / name):
            if len(examples) == limit:
                break
            identifier = row["id"]
            if identifier in prompts or identifier not in answers:
                raise ValueError(f"duplicate or unmatched BFCL ID: {identifier}")
            calls = answers[identifier]["ground_truth"]
            if len(calls) != 1 or len(calls[0]) != 1:
                raise ValueError(f"unsupported BFCL answer shape: {identifier}")
            tool_name, choices = next(iter(calls[0].items()))
            if not isinstance(choices, dict):
                raise ValueError(f"unsupported BFCL answer shape: {identifier}")
            if any(
                not isinstance(values, list) or not values
                for values in choices.values()
            ):
                raise ValueError(f"invalid BFCL alternatives: {identifier}")
            # Representative only: all original alternatives remain in ground_truth.
            arguments = {
                key: values[0] for key, values in choices.items() if values[0] != ""
            }
            turns = row["question"]
            if len(turns) != 1 or len(turns[0]) != 1 or turns[0][0]["role"] != "user":
                raise ValueError(f"unsupported BFCL question shape: {identifier}")
            definitions = []
            for item in row["function"]:
                parameters = item["parameters"].copy()
                if parameters["type"] == "dict":
                    parameters["type"] = "object"  # BFCL's Python type spelling.
                definitions.append(
                    ToolDefinition(item["name"], item["description"], parameters)
                )
            examples.append(
                EvaluationExample(
                    id=identifier,
                    source=f"ShishirPatil/gorilla@{revision}",
                    language="en",
                    domain=category,
                    user_utterance=turns[0][0]["content"],
                    available_tools=tuple(definitions),
                    should_call_tool=True,
                    expected_tool_name=tool_name,
                    expected_arguments=arguments,
                    tags=("bfcl-v4", "ast", "representative-answer"),
                )
            )
            ground_truth[identifier] = answers[identifier]
            prompts[identifier] = row
    if len(examples) < limit:
        raise ValueError(f"requested {limit} BFCL examples, found {len(examples)}")
    return BfclBatch(tuple(examples), ground_truth, prompts)
