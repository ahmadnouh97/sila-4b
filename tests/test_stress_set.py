import json
from copy import deepcopy
from pathlib import Path

import pytest

from sila.validate_stress_set import validate_stress_set

TEMPLATE = Path(__file__).resolve().parents[1] / "data/stress_test.template.jsonl"


def _rows():
    return [
        json.loads(line) for line in TEMPLATE.read_text(encoding="utf-8").splitlines()
    ]


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _add_competitor(row):
    if row["should_call_tool"] and len(row["available_tools"]) == 1:
        other = deepcopy(row["available_tools"][0])
        other["name"] = f"other_{row['id']}"
        row["available_tools"].append(other)


def test_illustrative_template_is_valid():
    validate_stress_set(TEMPLATE)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows[1].update(id=rows[0]["id"]), "duplicate ID"),
        (
            lambda rows: rows[1].update(user_utterance=rows[0]["user_utterance"]),
            "duplicate user_utterance",
        ),
        (lambda rows: rows.pop(), "expected 1 per category"),
        (lambda rows: rows[0].update(dialect="unknown"), "unknown dialect"),
        (lambda rows: rows[0].update(user_utterance="weather?"), "Arabic text"),
        (lambda rows: rows[0].update(expected_tool_name="other"), "available_tools"),
        (
            lambda rows: rows[0].update(expected_arguments={"unknown": 1}),
            "not permitted",
        ),
        (lambda rows: rows[0].update(expected_arguments={}), "omit required"),
        (lambda rows: rows[0].update(should_call_tool=False), "negative examples"),
        (
            lambda rows: rows[7].update(
                should_call_tool=True,
                expected_tool_name="send_message",
                expected_arguments={"recipient": "سارة", "text": "مرحبا"},
            ),
            "positive/negative category",
        ),
        (
            lambda rows: rows[0]["available_tools"][0]["parameters"].update(
                required=["unknown"]
            ),
            "required keys",
        ),
        (
            lambda rows: rows[0]["available_tools"][0]["parameters"][
                "properties"
            ].update(city={"type": "mystery"}),
            "schema type",
        ),
        (
            lambda rows: rows[0].update(expected_arguments={"city": 123}),
            "violates tool schema",
        ),
    ],
)
def test_rejects_invalid_rows(tmp_path, change, message):
    rows = deepcopy(_rows())
    change(rows)
    path = tmp_path / "stress.jsonl"
    _write(path, rows)
    with pytest.raises(ValueError, match=message):
        validate_stress_set(path)


def test_rejects_unreviewed_final_set(tmp_path):
    rows = _rows()
    path = tmp_path / "stress.jsonl"
    _write(path, rows)
    with pytest.raises(ValueError, match="native_speaker_reviewed"):
        validate_stress_set(path, final=True)
    for row in rows:
        row["review_status"] = "native_speaker_reviewed"
        row["language"] = "ar"
        row["reference_date"] = "2026-09-22"
        _add_competitor(row)
    _write(path, rows)
    with pytest.raises(ValueError, match="expected 10 per category"):
        validate_stress_set(path, final=True)


def test_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "stress.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(UnicodeDecodeError):
        validate_stress_set(path)


def test_pilot_mode_checks_counts_and_fixed_date(tmp_path):
    rows = []
    for copy_number in range(3):
        for template in _rows():
            row = deepcopy(template)
            row["id"] = f"{row['id']}-{copy_number}"
            row["user_utterance"] += f" {copy_number}"
            row["language"] = "ar"
            row["reference_date"] = "2026-09-22"
            row["review_status"] = "pilot_draft_not_gold"
            _add_competitor(row)
            rows.append(row)
    path = tmp_path / "pilot.jsonl"
    _write(path, rows)
    assert sum(validate_stress_set(path, pilot=True).values()) == 30
    competitor = rows[0]["available_tools"].pop()
    _write(path, rows)
    with pytest.raises(ValueError, match="competing available tools"):
        validate_stress_set(path, pilot=True)
    rows[0]["available_tools"].append(competitor)
    utterance = rows[0]["user_utterance"]
    for word in ("بكرة", "بكرا"):
        rows[0]["user_utterance"] = f"أريد الخدمة {word}."
        _write(path, rows)
        with pytest.raises(ValueError, match="relative-date utterance"):
            validate_stress_set(path, pilot=True)
    rows[0]["user_utterance"] = utterance
    rows[0]["reference_date"] = "2026-09-23"
    _write(path, rows)
    with pytest.raises(ValueError, match="reference_date must be fixed"):
        validate_stress_set(path, pilot=True)


def test_draft_mode_requires_100_cases_without_claiming_review(tmp_path):
    rows = []
    for copy_number in range(10):
        for template in _rows():
            row = deepcopy(template)
            row["id"] = f"{row['id']}-{copy_number}"
            row["user_utterance"] += f" {copy_number}"
            row["language"] = "ar"
            row["reference_date"] = "2026-09-22"
            row["review_status"] = "draft_not_gold"
            _add_competitor(row)
            rows.append(row)
    path = tmp_path / "draft.jsonl"
    _write(path, rows)
    assert sum(validate_stress_set(path, draft=True).values()) == 100
    with pytest.raises(ValueError, match="native_speaker_reviewed"):
        validate_stress_set(path, final=True)
