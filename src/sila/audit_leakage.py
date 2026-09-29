"""Audit local evaluation prompts, tools, schemas, and targets against training data."""

import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from sila.benchmarks import load_arabfuncbench, load_bfcl

ROOT = Path.cwd()
ARABFUNCBENCH_REVISION = "c11e4e5ede503892b85633c40c16720c75310e48"
BFCL_REVISION = "9d8416a96d1d69975493f1b6d60ff07d12a1726a"


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_prompt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = (
        " " if unicodedata.category(char).startswith(("M", "P", "S")) else char
        for char in normalized
    )
    return " ".join("".join(characters).split())


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _tool_parts(tool: object) -> tuple[str, str, set[str]]:
    if isinstance(tool, dict):
        name = tool["name"]
        description = tool["description"]
        parameters = tool["parameters"]
    else:
        name = tool.name
        description = tool.description
        parameters = tool.parameters
    schema = {"name": name, "description": description, "parameters": parameters}
    properties = parameters.get("properties", {})
    return name, _canonical(schema), set(properties)


def _tool_sets(rows: list[dict]) -> tuple[set[str], set[str], set[str]]:
    names: set[str] = set()
    schemas: set[str] = set()
    properties: set[str] = set()
    for row in rows:
        for tool in row["available_tools"]:
            name, schema, keys = _tool_parts(tool)
            names.add(name)
            schemas.add(schema)
            properties.update(keys)
    return names, schemas, properties


def _target_set(rows: list[dict]) -> set[tuple[str, str]]:
    return {
        (row["expected_tool_name"], _canonical(row["expected_arguments"]))
        for row in rows
        if row.get("should_call_tool") and row.get("expected_tool_name")
    }


def _matches_bfcl_target(
    tool_name: str, arguments: dict, answer: dict
) -> bool:
    calls = answer["ground_truth"]
    if len(calls) != 1 or tool_name not in calls[0]:
        return False
    choices = calls[0][tool_name]
    return (
        set(arguments) <= choices.keys()
        and all(arguments[key] in choices[key] for key in arguments)
        and all(key in arguments or "" in values for key, values in choices.items())
    )


def _audit_group(
    name: str,
    eval_rows: list[dict],
    train_rows: list[dict],
    train_prompt_exact: set[str],
    train_prompt_normalized: set[str],
    train_tool_names: set[str],
    train_tool_schemas: set[str],
    train_properties: set[str],
    train_targets: set[tuple[str, str]],
    bfcl_answers: dict[str, dict] | None = None,
) -> dict:
    exact_prompts = []
    normalized_prompts = []
    target_overlaps = []
    eval_tool_names, eval_schemas, eval_properties = _tool_sets(eval_rows)
    for row in eval_rows:
        prompt = row["user_utterance"]
        if prompt in train_prompt_exact:
            exact_prompts.append(row["id"])
        if _normalize_prompt(prompt) in train_prompt_normalized:
            normalized_prompts.append(row["id"])
        if bfcl_answers is None:
            target = (row.get("expected_tool_name"), _canonical(row.get("expected_arguments")))
            if target in train_targets:
                target_overlaps.append(row["id"])
        elif row.get("expected_tool_name"):
            train_args_by_tool = defaultdict(list)
            for tool, args in train_targets:
                train_args_by_tool[tool].append(json.loads(args))
            answer = bfcl_answers[row["id"]]
            if any(
                _matches_bfcl_target(row["expected_tool_name"], args, answer)
                for args in train_args_by_tool[row["expected_tool_name"]]
            ):
                target_overlaps.append(row["id"])
    tool_name_overlap = sorted(train_tool_names & eval_tool_names)
    schema_overlap = len(train_tool_schemas & eval_schemas)
    property_overlap = sorted(train_properties & eval_properties)
    blockers = {
        "exact_prompts": exact_prompts,
        "normalized_prompts": normalized_prompts,
        "tool_names": tool_name_overlap,
        "complete_tool_schemas": schema_overlap,
        "expected_calls": target_overlaps,
    }
    return {
        "examples": len(eval_rows),
        "blockers": blockers,
        "property_name_overlap_diagnostic": property_overlap,
        "passed": not any(blockers.values()),
    }


def run_audit() -> dict:
    train_path = ROOT / "data" / "training.train.jsonl"
    validation_path = ROOT / "data" / "training.validation.jsonl"
    arabic_path = ROOT / "data" / "stress_test.jsonl"
    arabfunc_root = ROOT / "data" / "ArabFuncBench"
    bfcl_root = ROOT / "data" / "benchmarks" / "bfcl_v4"
    train_rows = _jsonl(train_path)
    validation_rows = _jsonl(validation_path)
    corpus_rows = train_rows + validation_rows
    arabic_rows = _jsonl(arabic_path)
    arabfunc_rows = list(
        load_arabfuncbench(
            arabfunc_root / "arab_func_bench_examples.json",
            arabfunc_root / "arab_func_bench_tools.json",
            revision=ARABFUNCBENCH_REVISION,
        )
    )
    bfcl_batch = load_bfcl(
        bfcl_root, revision=BFCL_REVISION, limit=500
    )
    bfcl_rows = list(bfcl_batch.examples)
    corpus_prompt_exact = {row["user_utterance"] for row in corpus_rows}
    corpus_prompt_normalized = {
        _normalize_prompt(row["user_utterance"]) for row in corpus_rows
    }
    corpus_tool_names, corpus_schemas, corpus_properties = _tool_sets(corpus_rows)
    corpus_targets = _target_set(corpus_rows)

    def as_dict(example: object) -> dict:
        return {
            "id": example.id,
            "user_utterance": example.user_utterance,
            "available_tools": list(example.available_tools),
            "should_call_tool": example.should_call_tool,
            "expected_tool_name": example.expected_tool_name,
            "expected_arguments": example.expected_arguments,
        }

    groups = {
        "arabic_stress_set": _audit_group(
            "arabic_stress_set",
            arabic_rows,
            corpus_rows,
            corpus_prompt_exact,
            corpus_prompt_normalized,
            corpus_tool_names,
            corpus_schemas,
            corpus_properties,
            corpus_targets,
        ),
        "arabfuncbench": _audit_group(
            "arabfuncbench",
            [as_dict(example) for example in arabfunc_rows],
            corpus_rows,
            corpus_prompt_exact,
            corpus_prompt_normalized,
            corpus_tool_names,
            corpus_schemas,
            corpus_properties,
            corpus_targets,
        ),
        "bfcl_v4": _audit_group(
            "bfcl_v4",
            [as_dict(example) for example in bfcl_rows],
            corpus_rows,
            corpus_prompt_exact,
            corpus_prompt_normalized,
            corpus_tool_names,
            corpus_schemas,
            corpus_properties,
            corpus_targets,
            bfcl_batch.ground_truth,
        ),
    }
    report = {
        "status": "passed" if all(group["passed"] for group in groups.values()) else "blocked",
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
        "arabic_eval_sha256": _sha256(arabic_path),
        "arabfuncbench_revision": ARABFUNCBENCH_REVISION,
        "arabfuncbench_metric_revision": "9d314f34b6cb54d2916e4e2b454b87911df3110e",
        "bfcl_revision": BFCL_REVISION,
        "bfcl_file_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted(bfcl_root.rglob("*.json"))
        },
        "arabfuncbench_file_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted(arabfunc_root.glob("arab_func_bench_*.json"))
        },
        "training_rows": {"train": len(train_rows), "validation": len(validation_rows)},
        "evaluation_groups": groups,
        "method": "Exact prompts, NFKC/casefold/diacritic/punctuation-normalized prompts, tool names, full schemas, and expected calls; BFCL targets check every permitted value and optional argument.",
        "limitations": "String and structure overlap checks do not rule out semantic similarity.",
    }
    output = ROOT / "reports" / "leakage_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_audit()
    print(f"Leakage audit {report['status']}: {report['evaluation_groups']}")


if __name__ == "__main__":
    main()
