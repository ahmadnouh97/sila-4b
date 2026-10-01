import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("occupied", [False, True])
def test_notebook_refuses_to_overwrite_a_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, occupied: bool
) -> None:
    notebook = Path(__file__).parents[1] / "notebooks/train_qwen_qlora.ipynb"
    setup = "".join(
        json.loads(notebook.read_text(encoding="utf-8"))["cells"][1]["source"]
    )
    (tmp_path / "README.md").touch()
    output = tmp_path / "outputs/qwen3-4b-arabic-qlora-v1.8"
    output.mkdir(parents=True)
    if occupied:
        (output / "run.json").write_text("saved run", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    if occupied:
        with pytest.raises(RuntimeError, match="new empty output directory"):
            exec(setup, {})
        assert (output / "run.json").read_text(encoding="utf-8") == "saved run"
    else:
        exec(setup, {})
        assert not list(output.iterdir())


def test_followup_training_requires_explicit_authorization() -> None:
    notebook = Path(__file__).parents[1] / "notebooks/train_qwen_qlora.ipynb"
    training = "".join(
        json.loads(notebook.read_text(encoding="utf-8"))["cells"][11]["source"]
    )
    with pytest.raises(RuntimeError, match="explicitly authorized"):
        exec(training, {"TRAINING_AUTHORIZED": False})
