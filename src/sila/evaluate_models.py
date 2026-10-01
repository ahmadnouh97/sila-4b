"""Paired, resumable NF4 evaluation of the pinned Qwen base and saved adapter."""

import argparse
import hashlib
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sila.audit_leakage import ARABFUNCBENCH_REVISION, BFCL_REVISION
from sila.benchmark_metrics import (
    ARAB_METRIC_REVISION,
    SOURCES,
    load_arab_metrics,
    load_bfcl_checker,
    prepare_metrics,
    score_arabfuncbench,
    score_bfcl,
    verified_source,
)
from sila.benchmarks import load_arabfuncbench, load_bfcl
from sila.evaluation import score_examples
from sila.schemas import EvaluationExample, RawPrediction, ToolDefinition
from sila.validate_stress_set import validate_stress_set

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PROMPT_VERSION = "qwen-native-tools-v1"
SUITE_COUNTS = {"arabic": 100, "bfcl": 500, "arabfuncbench": 1000}


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ensure_run_record(path: Path, settings: dict) -> None:
    """Refuse to mix predictions from different inputs, weights, or settings."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != settings:
            raise ValueError("output directory contains a different run; use a new one")
    else:
        write_json(path, settings)


def render_prompt(tokenizer: Any, example: EvaluationExample) -> str:
    """Render only the user prompt and available tools, never expected labels."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": example.user_utterance}],
        tools=[
            {"type": "function", "function": asdict(tool)}
            for tool in example.available_tools
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def comparison(
    base: dict,
    adapter: dict,
    bfcl_base: float | None,
    bfcl_adapter: float | None,
    *,
    complete: bool,
) -> dict:
    gain = adapter["exact_tool_call_accuracy"] - base["exact_tool_call_accuracy"]
    base_error = 1 - base["exact_tool_call_accuracy"]
    relative = gain / base_error if base_error else None
    english_change = (
        bfcl_adapter - bfcl_base
        if bfcl_base is not None and bfcl_adapter is not None
        else None
    )
    criteria = {
        "arabic_improvement": gain >= 0.05 - 1e-12
        or (relative is not None and relative >= 0.25 - 1e-12),
        "english_preservation": (
            english_change >= -0.02 - 1e-12 if english_change is not None else None
        ),
        "arabic_no_call_preservation": (
            adapter["negative_case_accuracy"] >= base["negative_case_accuracy"]
        ),
    }
    return {
        "arabic_call_change_percentage_points": 100 * gain,
        "arabic_relative_error_reduction": relative,
        "arabic_no_call_change_percentage_points": 100
        * (adapter["negative_case_accuracy"] - base["negative_case_accuracy"]),
        "english_change_percentage_points": (
            100 * english_change if english_change is not None else None
        ),
        "criteria": criteria,
        "success": all(criteria.values())
        if complete and english_change is not None
        else None,
    }


def load_inputs(
    root: Path,
    run_dir: Path,
    *,
    data_dir: Path | None = None,
    audit_path: Path | None = None,
) -> tuple[dict, dict, Any]:
    """Validate the unchanged reviewed set and all hashes from the training audit."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data_dir = data_dir or (root / run["training_manifest"]).parent
    audit_path = audit_path or root / run["leakage_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (run["model_id"], run["model_revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("training run does not use the experiment's pinned base")
    if audit["status"] != "passed":
        raise ValueError("leakage audit has not passed")
    for key, relative_path in (
        ("train_sha256", data_dir / "training.train.jsonl"),
        ("validation_sha256", data_dir / "training.validation.jsonl"),
        ("arabic_eval_sha256", "data/stress_test.jsonl"),
    ):
        if sha256(root / relative_path) != run[key] or run[key] != audit[key]:
            raise ValueError(f"training/audit hash mismatch: {relative_path}")
    expected_revisions = {
        "arabfuncbench_revision": ARABFUNCBENCH_REVISION,
        "bfcl_revision": BFCL_REVISION,
        "arabfuncbench_metric_revision": ARAB_METRIC_REVISION,
    }
    if any(audit[key] != value for key, value in expected_revisions.items()):
        raise ValueError("benchmark audit revisions do not match the experiment")
    for group in ("bfcl_file_sha256", "arabfuncbench_file_sha256"):
        for relative_path, expected_hash in audit[group].items():
            if sha256(root / relative_path) != expected_hash:
                raise ValueError(f"benchmark audit hash mismatch: {relative_path}")
    validate_stress_set(root / "data/stress_test.jsonl", final=True)
    rows = [
        json.loads(line)
        for line in (root / "data/stress_test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    arabic = tuple(
        EvaluationExample(
            id=row["id"],
            source="manual_arabic_stress_set",
            language=row["language"],
            domain=row["category"],
            user_utterance=row["user_utterance"],
            available_tools=tuple(
                ToolDefinition(**tool) for tool in row["available_tools"]
            ),
            should_call_tool=row["should_call_tool"],
            expected_tool_name=row["expected_tool_name"],
            expected_arguments=row["expected_arguments"],
            dialect=row["dialect"],
        )
        for row in rows
    )
    bfcl = load_bfcl(root / "data/benchmarks/bfcl_v4", revision=BFCL_REVISION)
    if sum(example.domain == "simple_python" for example in bfcl.examples) != 400:
        raise ValueError("BFCL subset must contain 400 simple_python and 100 multiple")
    external = load_arabfuncbench(
        root / "data/ArabFuncBench/arab_func_bench_examples.json",
        root / "data/ArabFuncBench/arab_func_bench_tools.json",
        revision=ARABFUNCBENCH_REVISION,
    )
    suites = {"arabic": arabic, "bfcl": bfcl.examples, "arabfuncbench": external}
    if any(len(examples) != SUITE_COUNTS[name] for name, examples in suites.items()):
        raise ValueError("evaluation set counts differ from the fixed experiment")
    return run, suites, bfcl


def read_predictions(
    path: Path, examples: tuple, variant: str, generation: dict
) -> list[RawPrediction]:
    if not path.exists():
        return []
    predictions = []
    # ponytail: reject partial JSONL tails; recover the fragment before resuming.
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        predictions.append(
            RawPrediction(
                **{key: row[key] for key in RawPrediction.__dataclass_fields__}
            )
        )
    identifiers = [prediction.example_id for prediction in predictions]
    if identifiers != [example.id for example in examples[: len(predictions)]]:
        raise ValueError(f"prediction IDs are not an unchanged prefix: {path}")
    model_id = MODEL_ID if variant == "base" else MODEL_ID + "+arabic-qlora"
    if any(
        prediction.model_id != model_id
        or prediction.model_revision != MODEL_REVISION
        or prediction.prompt_version != PROMPT_VERSION
        or prediction.generation_config != generation
        for prediction in predictions
    ):
        raise ValueError(f"saved prediction metadata differs from this run: {path}")
    return predictions


def generate_predictions(
    model: Any,
    tokenizer: Any,
    examples: tuple,
    path: Path,
    variant: str,
    generation: dict,
    existing: list[RawPrediction],
) -> list[RawPrediction]:
    import torch

    predictions = list(existing)
    context = (
        model.disable_adapter()
        if variant == "base" and hasattr(model, "disable_adapter")
        else nullcontext()
    )
    with context, torch.inference_mode(), path.open("a", encoding="utf-8") as file:
        for index, example in enumerate(examples[len(existing) :], len(existing) + 1):
            prompt = render_prompt(tokenizer, example)
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            input_tokens = inputs["input_ids"].shape[1]
            if (
                input_tokens + generation["max_new_tokens"]
                > model.config.max_position_embeddings
            ):
                raise ValueError(
                    f"context limit exceeded for {example.id}; no truncation"
                )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.generate(
                **inputs,
                **generation,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            generated = output[0, input_tokens:]
            raw = RawPrediction(
                example_id=example.id,
                generated_text=tokenizer.decode(generated, skip_special_tokens=True),
                model_id=MODEL_ID if variant == "base" else MODEL_ID + "+arabic-qlora",
                model_revision=MODEL_REVISION,
                prompt_version=PROMPT_VERSION,
                generation_config=generation,
                latency_seconds=time.perf_counter() - started,
                gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
            )
            row = {
                **asdict(raw),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_tokens": input_tokens,
                "output_tokens": len(generated),
                "hit_token_limit": len(generated) == generation["max_new_tokens"]
                and int(generated[-1]) != tokenizer.eos_token_id,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
            predictions.append(raw)
            print(
                f"{path.stem}: {index}/{len(examples)} {example.id} "
                f"({raw.latency_seconds:.1f}s, {len(generated)} tokens)",
                flush=True,
            )
    return predictions


def summarize(
    root: Path, output: Path, suites: dict, predictions: dict, bfcl: Any, vendor: Path
) -> dict:
    summary = {"evaluations": {}, "verdict": None}
    bfcl_namespace = load_bfcl_checker(vendor) if "bfcl" in suites else None
    arab_rows = []
    arab_namespace = None
    if "arabfuncbench" in suites:
        arab_rows = json.loads(
            (root / "data/ArabFuncBench/arab_func_bench_examples.json").read_text(
                encoding="utf-8"
            )
        )
        tools = json.loads(
            (root / "data/ArabFuncBench/arab_func_bench_tools.json").read_text(
                encoding="utf-8"
            )
        )
        arab_namespace = load_arab_metrics(vendor, tools)
    reports = {}
    for suite, examples in suites.items():
        variants = {}
        for variant in ("base", "adapter"):
            raw = predictions[(suite, variant)]
            report = score_examples(examples, raw, output_format="qwen")
            reports[(suite, variant)] = report
            if suite == "bfcl":
                scored = score_bfcl(bfcl, report, bfcl_namespace)
            elif suite == "arabfuncbench":
                scored = score_arabfuncbench(arab_rows, report, arab_namespace)
            else:
                scored = asdict(report)
            write_json(output / f"{suite}.{variant}.scores.json", scored)
            variants[variant] = {
                key: value
                for key, value in scored.items()
                if key not in {"examples", "results"}
            }
            variants[variant]["inference"] = {
                "generation_seconds": sum(item.latency_seconds for item in raw),
                "peak_allocated_gpu_mb": max(item.gpu_memory_mb or 0 for item in raw),
            }
            if suite == "arabic":
                variants[variant]["categories"] = {
                    category: {
                        "correct": sum(
                            item.result.exact_match
                            for example, item in zip(examples, report.examples)
                            if example.domain == category
                        ),
                        "total": sum(
                            example.domain == category for example in examples
                        ),
                    }
                    for category in dict.fromkeys(
                        example.domain for example in examples
                    )
                }
        summary["evaluations"][suite] = {
            "complete": len(examples) == SUITE_COUNTS[suite],
            **variants,
        }
    if "arabic" in suites:
        english = summary["evaluations"].get("bfcl")
        summary["verdict"] = comparison(
            reports[("arabic", "base")].rates,
            reports[("arabic", "adapter")].rates,
            english["base"]["accuracy"] if english else None,
            english["adapter"]["accuracy"] if english else None,
            complete=len(suites["arabic"]) == 100
            and english is not None
            and english["complete"],
        )
        paired = []
        for example, base, adapter in zip(
            suites["arabic"],
            reports[("arabic", "base")].examples,
            reports[("arabic", "adapter")].examples,
        ):
            paired.append(
                {
                    "example": asdict(example),
                    "base": asdict(base.result),
                    "adapter": asdict(adapter.result),
                }
            )
        write_json(output / "arabic.paired.json", paired)
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("outputs/qwen3-4b-arabic-qlora")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/evaluation/results")
    )
    parser.add_argument("--suite", choices=["all", *SUITE_COUNTS], default="all")
    parser.add_argument(
        "--limit", type=int, help="smoke test only; cannot produce a final verdict"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--adapter-dir", type=Path, help="saved checkpoint to evaluate")
    parser.add_argument(
        "--prepare-metrics",
        action="store_true",
        help="download pinned metric sources only",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="score complete saved predictions without loading weights",
    )
    args = parser.parse_args()
    root = Path.cwd()
    vendor = root / "outputs/evaluation/vendor"
    if args.prepare_metrics:
        prepare_metrics(vendor)
        print(f"Pinned evaluator sources ready: {vendor}")
        return
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("token and example limits must be positive")
    run, all_suites, bfcl = load_inputs(
        root,
        args.run_dir,
        data_dir=args.data_dir,
        audit_path=args.audit_path,
    )
    suites = {
        name: examples[: args.limit] if args.limit else examples
        for name, examples in all_suites.items()
        if args.suite in {"all", name}
    }
    if any(name != "arabic" for name in suites):
        for name in SOURCES:
            verified_source(vendor, name)
    adapter = args.adapter_dir or args.run_dir / "adapter"
    generation = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    settings = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "training_run_sha256": sha256(args.run_dir / "run.json"),
        "runner_sha256": sha256(Path(__file__)),
        "training_metrics": run["train_metrics"],
        "prompt_version": PROMPT_VERSION,
        "quantization": {"type": "nf4", "double_quant": True},
        "generation_config": generation,
        "example_ids": {
            name: [example.id for example in examples]
            for name, examples in suites.items()
        },
        "data_hashes": {
            key: run[key]
            for key in ("train_sha256", "validation_sha256", "arabic_eval_sha256")
        },
        "audit_sha256": sha256(args.audit_path or root / run["leakage_audit"]),
        "metric_sources": {name: expected for name, (_, expected) in SOURCES.items()},
        "software": {
            name: version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = args.output_dir / "run.json"
    if not record.exists() and any(args.output_dir.glob("*.raw.jsonl")):
        raise ValueError(
            "raw predictions exist without a run record; use a new output directory"
        )
    ensure_run_record(record, settings)
    predictions = {
        (suite, variant): read_predictions(
            args.output_dir / f"{suite}.{variant}.raw.jsonl",
            examples,
            variant,
            generation,
        )
        for suite, examples in suites.items()
        for variant in ("base", "adapter")
    }
    pending = any(
        len(predictions[(suite, variant)]) < len(examples)
        for suite, examples in suites.items()
        for variant in ("base", "adapter")
    )
    if args.score_only and pending:
        raise ValueError("predictions are incomplete; resume inference before scoring")
    if pending:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for paired NF4 evaluation")
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError(
                "less than 4 GiB GPU memory free; shut down the training kernel first"
            )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        set_seed(42)
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
                tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
        }
        ensure_run_record(args.output_dir / "hardware.json", hardware)
        base = AutoModelForCausalLM.from_pretrained(
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
        model = PeftModel.from_pretrained(
            base, adapter, is_trainable=False, local_files_only=True
        )
        model.eval()
        for suite, examples in suites.items():
            for variant in ("base", "adapter"):
                predictions[(suite, variant)] = generate_predictions(
                    model,
                    tokenizer,
                    examples,
                    args.output_dir / f"{suite}.{variant}.raw.jsonl",
                    variant,
                    generation,
                    predictions[(suite, variant)],
                )
        del model, base
        torch.cuda.empty_cache()
    summary = summarize(root, args.output_dir, suites, predictions, bfcl, vendor)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Evaluation files: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
