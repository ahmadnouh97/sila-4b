"""Score independent development cases on the cached base or a saved checkpoint."""

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from sila.evaluate_models import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_VERSION,
    comparison,
    ensure_run_record,
    generate_predictions,
    read_predictions,
    sha256,
    write_json,
)
from sila.evaluation import score_examples
from sila.followup_data import to_example


def select_checkpoint(base: dict, candidates: dict[str, dict]) -> str | None:
    """Require development improvement and zero English/no-call regression."""
    eligible = {
        name: scores
        for name, scores in candidates.items()
        if comparison(
            base["ar"],
            scores["ar"],
            base["en"]["exact_tool_call_accuracy"],
            scores["en"]["exact_tool_call_accuracy"],
            complete=True,
        )["success"]
        and scores["en"]["exact_tool_call_accuracy"]
        >= base["en"]["exact_tool_call_accuracy"]
    }
    return (
        max(eligible, key=lambda name: eligible[name]["ar"]["exact_tool_call_accuracy"])
        if eligible
        else None
    )


def verified_checkpoint(results_dir: Path) -> dict:
    """Refuse to promote checkpoint files changed since their responses were scored."""
    record = json.loads((results_dir / "run.json").read_text(encoding="utf-8"))
    if not record.get("adapter"):
        raise ValueError("candidate must refer to a scored adapter")
    adapter = Path(record["adapter"])
    for filename, key in (
        ("adapter_model.safetensors", "adapter_sha256"),
        ("adapter_config.json", "adapter_config_sha256"),
    ):
        if sha256(adapter / filename) != record[key]:
            raise ValueError("checkpoint changed since development scoring")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/followup-v1.8/development.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--select",
        type=Path,
        nargs="+",
        help="candidate score directories; output-dir is baseline",
    )
    args = parser.parse_args()
    if args.select:
        base = json.loads(
            (args.output_dir / "summary.json").read_text(encoding="utf-8")
        )
        candidates = {}
        records = {}
        for path in args.select:
            candidate = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            record = verified_checkpoint(path)
            if candidate["adapter"] != record["adapter"]:
                raise ValueError("summary does not match the scored checkpoint")
            if candidate["settings"] != base["settings"]:
                raise ValueError(
                    "development inputs, prompts, software, or decoding differ"
                )
            if json.loads((path / "hardware.json").read_text()) != json.loads(
                (args.output_dir / "hardware.json").read_text()
            ):
                raise ValueError(
                    "development hardware, quantization dtype, or template differ"
                )
            candidates[str(path)] = candidate
            records[str(path)] = record
        selected = select_checkpoint(base, candidates)
        result = {
            "selected_results": selected,
            "selected_adapter": candidates[selected]["adapter"] if selected else None,
            "promote_to_final_evaluation": selected is not None,
            "selected_weight_hash": records[selected]["adapter_sha256"]
            if selected
            else None,
            "selected_config_hash": records[selected]["adapter_config_sha256"]
            if selected
            else None,
        }
        write_json(args.output_dir / "selection.json", result)
        print(json.dumps(result, indent=2))
        return
    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row.get("review_status") != "ai_reviewed" for row in rows):
        raise ValueError("development cases require recorded AI review before scoring")
    examples = tuple(to_example(row) for row in rows)
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("duplicate development IDs")
    for language in ("ar", "en"):
        subset = [e for e in examples if e.language == language]
        if not any(e.should_call_tool for e in subset) or not any(
            not e.should_call_tool for e in subset
        ):
            raise ValueError("both languages need call and no-call development cases")
    generation = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
        "use_cache": True,
    }
    settings = {
        "data_sha256": sha256(args.data),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "generation": generation,
        "runner_sha256": sha256(Path(__file__)),
        "inference_sha256": sha256(Path(__file__).with_name("evaluate_models.py")),
        "software": {
            name: version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes")
        },
    }
    record = {
        **settings,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
    }
    if args.adapter:
        record["adapter_sha256"] = sha256(args.adapter / "adapter_model.safetensors")
        record["adapter_config_sha256"] = sha256(args.adapter / "adapter_config.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run.json"
    if not run_path.exists() and (args.output_dir / "responses.raw.jsonl").exists():
        raise ValueError("raw predictions exist without a run record")
    ensure_run_record(run_path, record)
    variant = "adapter" if args.adapter else "base"
    raw_path = args.output_dir / "responses.raw.jsonl"
    predictions = read_predictions(raw_path, examples, variant, generation)
    if len(predictions) < len(examples):
        if args.score_only:
            raise ValueError("incomplete predictions; cannot score")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4 * 1024**3:
            raise RuntimeError("CUDA with at least 4 GiB free is required")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        hardware = {
            "gpu": torch.cuda.get_device_name(0),
            "compute_dtype": str(dtype),
            "cuda": torch.version.cuda,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode()
            ).hexdigest(),
        }
        ensure_run_record(args.output_dir / "hardware.json", hardware)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
        )
        if args.adapter:
            model = PeftModel.from_pretrained(
                model, args.adapter, is_trainable=False, local_files_only=True
            )
        model.eval()
        predictions = generate_predictions(
            model, tokenizer, examples, raw_path, variant, generation, predictions
        )
        del model
        torch.cuda.empty_cache()
    summary = {"settings": settings, "adapter": record["adapter"]}
    for language in ("ar", "en"):
        selected = [i for i, e in enumerate(examples) if e.language == language]
        report = score_examples(
            [examples[i] for i in selected],
            [predictions[i] for i in selected],
            output_format="qwen",
        )
        write_json(args.output_dir / f"{language}.scores.json", asdict(report))
        summary[language] = report.rates
        summary[f"{language}_counts"] = report.counts
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
