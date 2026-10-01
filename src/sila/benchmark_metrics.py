"""Load hash-verified upstream metrics without their inference clients.

Only reviewed, pinned evaluator source is compiled here. Model responses are
parsed as data by sila.evaluation; they are never compiled or executed.
"""

import ast
import hashlib
import json
import re
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

from sila.audit_leakage import BFCL_REVISION
from sila.benchmarks import BfclBatch
from sila.evaluation import ScoreReport

ARAB_METRIC_REVISION = "9d314f34b6cb54d2916e4e2b454b87911df3110e"
BFCL_URL = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_REVISION}/"
    "berkeley-function-call-leaderboard/bfcl_eval/"
)
SOURCES = {
    "bfcl_eval/constants/enums.py": (
        BFCL_URL + "constants/enums.py",
        "2182becfa2a1d071ee1db30db593b4758c6bf866aa12d2d4b8daf09175ea518a",
    ),
    "bfcl_eval/constants/type_mappings.py": (
        BFCL_URL + "constants/type_mappings.py",
        "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd",
    ),
    "bfcl_eval/eval_checker/ast_eval/type_convertor/java_type_converter.py": (
        BFCL_URL + "eval_checker/ast_eval/type_convertor/java_type_converter.py",
        "2fd4f4b0443b3dd974a1723bb4e45c086d7b352631062da7807ad1ad40706604",
    ),
    "bfcl_eval/eval_checker/ast_eval/type_convertor/js_type_converter.py": (
        BFCL_URL + "eval_checker/ast_eval/type_convertor/js_type_converter.py",
        "a114e9ff75c025cb52787ac33d6c2fbaa390905c6125a2b3c6afebab232bb5e4",
    ),
    "bfcl_ast_checker.py": (
        BFCL_URL + "eval_checker/ast_eval/ast_checker.py",
        "2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538",
    ),
    "arab_evaluation.ipynb": (
        "https://raw.githubusercontent.com/lsadouk/ArabFuncBench/"
        f"{ARAB_METRIC_REVISION}/ArabFuncBench_Evaluation_final.ipynb",
        "73e96ae2b5c463b780459d94acadf460ffd9c9ef6825d60016fc793d966da71a",
    ),
}


def verified_source(vendor: Path, name: str) -> bytes:
    """Reject missing or modified upstream source before executing any of it."""
    data = (vendor / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCES[name][1]:
        raise ValueError(f"upstream metric source hash mismatch: {name}")
    return data


def prepare_metrics(vendor: Path) -> None:
    """Download only the six pinned scoring sources into ignored local storage."""
    for name, (url, expected_hash) in SOURCES.items():
        path = vendor / name
        if path.exists():
            verified_source(vendor, name)
            continue
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"downloaded metric source hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def load_bfcl_checker(vendor: Path) -> dict:
    """Keep official scoring functions intact; supply a Qwen naming registry.

    The registry only controls dot/underscore conversion. Our native prompts
    preserve original function names, so that conversion is disabled. No BFCL
    inference handler, API client, or generated Python code is loaded.
    """
    namespace = {
        "MODEL_CONFIG_MAPPING": {
            "Qwen/Qwen3-4B-Instruct-2507": SimpleNamespace(underscore_to_dot=False)
        }
    }
    for name in SOURCES:
        if not name.endswith(".py"):
            continue
        tree = ast.parse(verified_source(vendor, name), filename=name)
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("bfcl_eval")
            )
        ]
        exec(compile(tree, name, "exec"), namespace)
    return namespace


def load_arab_metrics(vendor: Path, tools: dict) -> dict:
    """Load only upstream metric definitions, excluding notebook side effects."""
    notebook = json.loads(verified_source(vendor, "arab_evaluation.ipynb"))
    namespace = {"re": re, "SequenceMatcher": SequenceMatcher, "all_tools": tools}
    names = {"char_similarity", "is_arabic", "evaluate_response", "compute_metrics"}
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def evaluate_response(" not in source:
            continue
        tree = ast.parse(source)
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        found.update(node.name for node in tree.body)
        exec(compile(tree, "arab_evaluation.ipynb", "exec"), namespace)
    if found != names:
        raise ValueError("upstream ArabFuncBench metric definitions missing")
    return namespace


def score_bfcl(batch: BfclBatch, report: ScoreReport, namespace: dict) -> dict:
    """Score native JSON calls against original BFCL schemas and alternatives."""
    results = []
    for item in report.examples:
        result = item.result
        call = result.parsed_call
        if result.parse_error or call is None:
            checked = {"valid": False, "error": [result.parse_error or "no call"]}
        else:
            checked = namespace["ast_checker"](
                deepcopy(batch.prompts[result.example_id]["function"]),
                [{call.tool_name: deepcopy(call.arguments)}],
                deepcopy(batch.ground_truth[result.example_id]["ground_truth"]),
                namespace["Language"].PYTHON,
                batch.prompts[result.example_id]["id"].rsplit("_", 1)[0],
                "Qwen/Qwen3-4B-Instruct-2507",
            )
        results.append({"id": result.example_id, **checked})
    correct = sum(result["valid"] for result in results)
    return {
        "correct": correct,
        "total": len(results),
        "accuracy": correct / len(results),
        "results": results,
    }


def score_arabfuncbench(rows: list[dict], report: ScoreReport, namespace: dict) -> dict:
    """Translate native calls to upstream metric inputs without repairing answers.

    Invalid structured output must fail even negative cases; the upstream JSON
    parser's malformed-to-null fallback is deliberately not used for Qwen text.
    """
    results = []
    # Pair by unchanged source order: the upstream export repeats ten IDs.
    for index, (row, item) in enumerate(
        zip(rows[: len(report.examples)], report.examples, strict=True)
    ):
        result = item.result
        call = result.parsed_call
        response = {
            "function_called": (
                object() if result.parse_error else call.tool_name if call else None
            ),
            "arguments": call.arguments if call else None,
        }
        results.append(
            {
                **namespace["evaluate_response"](row, response),
                "source_row_index": index,
                "prediction_id": result.example_id,
            }
        )
    return {"metrics": namespace["compute_metrics"](results), "results": results}
