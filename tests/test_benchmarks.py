import json

import pytest

from sila.benchmarks import load_arabfuncbench, load_bfcl


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_arabfuncbench_local_fixture(tmp_path):
    tools = tmp_path / "tools.json"
    rows = tmp_path / "examples.json"
    write_json(
        tools,
        [
            {
                "name": "weather",
                "description": "الطقس",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
    )
    write_json(
        rows,
        [
            {
                "id": "arab-1",
                "domain": "services",
                "utterance": "ما الطقس؟",
                "is_negative": False,
                "expected_function": "weather",
                "expected_arguments": {"city": "الرياض"},
                "available_tools": ["weather"],
            },
            {
                "id": "arab-2",
                "domain": "services",
                "utterance": "مرحبا",
                "is_negative": True,
                "expected_function": None,
                "expected_arguments": None,
                "available_tools": ["weather"],
            },
        ],
    )

    examples = load_arabfuncbench(rows, tools, revision="abc123")
    assert [item.id for item in examples] == ["arab-1", "arab-2"]
    assert examples[0].expected_arguments == {"city": "الرياض"}
    assert examples[1].should_call_tool is False
    assert all(item.source.endswith("@abc123") for item in examples)
    with pytest.raises(ValueError):
        load_arabfuncbench(rows, tools, revision="")


def test_bfcl_local_fixture_preserves_official_answers(tmp_path):
    answer_dir = tmp_path / "possible_answer"
    answer_dir.mkdir()
    for category, identifier, choices in (
        (
            "simple_python",
            "simple_python_0",
            {"city": ["Paris", "paris"], "unit": ["", "C"]},
        ),
        ("multiple", "multiple_0", {"city": ["London"]}),
    ):
        name = f"BFCL_v4_{category}.json"
        prompt = {
            "id": identifier,
            "question": [[{"role": "user", "content": "Weather?"}]],
            "function": [
                {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "dict",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
        answer = {"id": identifier, "ground_truth": [{"weather": choices}]}
        write_json(tmp_path / name, prompt)
        write_json(answer_dir / name, answer)

    batch = load_bfcl(tmp_path, revision="def456", limit=2)
    assert [item.id for item in batch.examples] == ["simple_python_0", "multiple_0"]
    assert batch.examples[0].expected_arguments == {"city": "Paris"}
    assert batch.examples[0].available_tools[0].parameters["type"] == "object"
    assert (
        batch.prompts["simple_python_0"]["function"][0]["parameters"]["type"] == "dict"
    )
    assert batch.ground_truth["simple_python_0"]["ground_truth"] == [
        {"weather": {"city": ["Paris", "paris"], "unit": ["", "C"]}}
    ]
    assert all(item.source.endswith("@def456") for item in batch.examples)
    (tmp_path / "BFCL_v4_multiple.json").unlink()
    (answer_dir / "BFCL_v4_multiple.json").unlink()
    assert len(load_bfcl(tmp_path, revision="def456", limit=1).examples) == 1
    with pytest.raises(ValueError, match="supported BFCL categories"):
        load_bfcl(tmp_path, revision="def456", categories=("parallel",))
