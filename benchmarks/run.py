import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

from ncm_engine.inference import Engine

PROMPTS = Path(__file__).with_name("prompts.json")


def summarize(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
    }


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def stage_series(records):
    keys = sorted({k for record in records for k in record.milliseconds})
    values = {
        key: summarize([r.milliseconds.get(key, 0) for r in records]) for key in keys
    }
    if any(k.startswith("transformer_level_") for k in keys):
        values["transformer"] = summarize(
            [
                sum(
                    v for k, v in r.milliseconds.items() if k.startswith("transformer_")
                )
                for r in records
            ]
        )
    return values


def signature(result):
    codes = json.dumps(result.concept_codes, separators=(",", ":")).encode()
    tokens = json.dumps(result.token_ids, separators=(",", ":")).encode()
    return {
        "concept_sha256": hashlib.sha256(codes).hexdigest(),
        "token_sha256": hashlib.sha256(tokens).hexdigest(),
        "text": result.text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run one profile in an isolated process."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/xsum"))
    parser.add_argument("--vocabulary", type=Path, default=Path("artifacts/gpt2"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--profile", default="balanced", choices=("quality", "balanced", "ultra")
    )
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(text, str) or not text.strip() for text in prompts)
    ):
        parser.error("--prompts must be a JSON array of nonempty strings")
    if args.runs < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("runs and threads must be positive and warmup must be nonnegative")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    engine = Engine(args.checkpoint, args.vocabulary, args.device, args.profile)
    first = engine.generate(prompts[0])
    first_ready = (perf_counter() - started) * 1000
    for i in range(args.warmup):
        engine.generate(prompts[i % len(prompts)])
    latencies = []
    for i in range(args.runs):
        torch.manual_seed(args.seed + i)
        latencies.append(
            engine.generate(prompts[i % len(prompts)]).milliseconds["total"]
        )
    steady_peak = peak_rss()
    detailed, reconstruction = [], []
    for i in range(args.runs):
        torch.manual_seed(args.seed + i)
        prompt = prompts[i % len(prompts)]
        detailed.append(engine.generate(prompt, detailed=True))
        reconstruction.append(engine.reconstruct(prompt, detailed=True))
    accelerator = {}
    if args.device == "cuda":
        accelerator["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if args.device == "mps":
        accelerator["mps_current_allocated_bytes"] = (
            torch.mps.current_allocated_memory()
        )
        accelerator["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    report = {
        "device": args.device,
        "profile": args.profile,
        "task": engine.config.task,
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": str(torch.__version__),
        "threads": args.threads,
        "seed": args.seed,
        "warmup": args.warmup,
        "runs": args.runs,
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "checkpoint_bytes": engine.checkpoint_bytes,
        "load_ms": engine.load_ms,
        "first_generation_ms": first.milliseconds["total"],
        "load_and_first_ms": first_ready,
        "steady_latency_ms": summarize(latencies),
        "requests_per_second": len(latencies) * 1000 / sum(latencies),
        "steady_peak_rss_bytes": steady_peak,
        "peak_rss_bytes": peak_rss(),
        **accelerator,
        "generation_stage_ms": stage_series(detailed),
        "reconstruction_stage_ms": stage_series(reconstruction),
        "generation": [signature(item) for item in detailed],
        "reconstruction": [signature(item) for item in reconstruction],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.profile}/{args.device}: {report['steady_latency_ms']}")


if __name__ == "__main__":
    main()
