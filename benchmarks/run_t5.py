"""Offline conventional encoder-decoder baseline, isolated from NCM runtime."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from benchmarks.run import PROMPTS, peak_rss, summarize
from ncm_engine.device import select_device, synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if not prompts or args.runs < 1 or args.warmup < 0 or args.threads < 1:
        parser.error(
            "Require prompts, positive runs and threads, and nonnegative warmup"
        )
    device = select_device(args.device)
    torch.set_num_threads(args.threads)
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            local_files_only=True,
            use_safetensors=True,
        )
        .eval()
        .to(device)
    )
    synchronize(device)
    load_ms = (perf_counter() - started) * 1000

    @torch.inference_mode()
    def generate(prompt):
        start = perf_counter()
        tokens = tokenizer(
            "summarize: " + prompt, max_length=512, truncation=True, return_tensors="pt"
        )
        output = model.generate(
            **tokens.to(device), max_new_tokens=48, do_sample=False, num_beams=1
        )
        answer = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)
        synchronize(device)
        return (perf_counter() - start) * 1000, answer

    first_ms, _ = generate(prompts[0])
    for i in range(args.warmup):
        generate(prompts[i % len(prompts)])
    results = [generate(prompts[i % len(prompts)]) for i in range(args.runs)]
    latencies = [ms for ms, _ in results]
    report = {
        "task": "xsum",
        "model": "google-t5/t5-small",
        "device": device.type,
        "threads": args.threads,
        "torch": str(torch.__version__),
        "os": platform.platform(),
        "runs": args.runs,
        "warmup": args.warmup,
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "checkpoint_bytes": sum(
            path.stat().st_size for path in args.model.glob("*.safetensors")
        ),
        "load_ms": load_ms,
        "first_generation_ms": first_ms,
        "steady_latency_ms": summarize(latencies),
        "peak_rss_bytes": peak_rss(),
        "requests_per_second": len(latencies) * 1000 / sum(latencies),
        "generation": [{"text": answer} for _, answer in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["steady_latency_ms"])


if __name__ == "__main__":
    main()
