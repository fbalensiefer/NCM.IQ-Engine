import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Compare compatible benchmark JSON reports."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline, candidate = (
        json.loads(p.read_text()) for p in (args.baseline, args.candidate)
    )
    for key in (
        "device",
        "threads",
        "seed",
        "prompts_sha256",
        "task",
        "runs",
        "warmup",
        "torch",
        "os",
    ):
        if baseline[key] != candidate[key]:
            parser.error(f"Comparison requires the same {key}")
    print("| Metric | Baseline ms | Candidate ms | Speedup |")
    print("|---|---:|---:|---:|")
    for label, stage, key in (
        ("Full generation", None, "steady_latency_ms"),
        ("Transformer", "generation_stage_ms", "transformer"),
        ("Decoder forward", "generation_stage_ms", "neural_decoder"),
        ("Decoder text selection", "generation_stage_ms", "decoder_selection"),
        ("Neural encoder", "reconstruction_stage_ms", "neural_encoder"),
        ("Exact RVQ search", "reconstruction_stage_ms", "rvq_search"),
        ("Full reconstruction", "reconstruction_stage_ms", "total"),
    ):
        a, b = (
            report[stage][key] if stage else report[key]
            for report in (baseline, candidate)
        )
        a, b = a["median"], b["median"]
        print(f"| {label} | {a:.3f} | {b:.3f} | {a / b:.2f}× |")
    for task in ("generation", "reconstruction"):
        rows = list(zip(baseline[task], candidate[task], strict=True))
        codes = sum(a["concept_sha256"] == b["concept_sha256"] for a, b in rows)
        tokens = sum(a["token_sha256"] == b["token_sha256"] for a, b in rows)
        print(
            f"{task}: {codes}/{len(rows)} identical concept arrays, {tokens}/{len(rows)} identical token arrays"
        )


if __name__ == "__main__":
    main()
