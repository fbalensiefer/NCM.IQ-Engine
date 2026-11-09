# NCM Chat

A local inference workbench and chat interface for [Next-Concept Modeling](https://arxiv.org/pdf/2610.14162v1). A multiscale residual VQ-VAE represents text as discrete code sequences. A Transformer predicts all positions of a scale in parallel, adds the resulting residual to the reconstruction, and proceeds to the next scale. A neural decoder maps the final latent to text. The application runs these models locally.

The official **XSum** checkpoint summarizes passages. Select a **dialogue-adapted** checkpoint to include previous turns in model context. The included training recipe demonstrates context-dependent replies on a tiny corpus. General dialogue quality depends on adaptation data and evaluation.

## Run locally

Use Python 3.12 on Apple Silicon macOS or Linux. PyTorch selects an available CUDA or MPS device automatically, with CPU as the fallback. CPU and MPS were tested on an Apple M2. The CUDA path uses PyTorch dispatch and awaits hardware validation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[download]'
ncm-download
HF_HUB_OFFLINE=1 ncm-chat
```

Open [the local interface](http://127.0.0.1:8000). The server binds to loopback and stores sessions in `sessions/chat.sqlite3`. For an existing checkout with populated `artifacts/xsum` and `artifacts/gpt2`, start at the launch command after installing. The 624 MiB official checkpoint pair and the text vocabulary are downloaded separately. Subsequent requests use local files.

| Profile | Behavior |
|---|---|
| Quality | FP32 upstream Transformer and RVQ inference with the released decoder policy. |
| Balanced | FP32 prefix K/V reuse, padded-prefix trimming, final latent reuse, cached exact RVQ norms and sparse decoder penalties. |
| Ultra-light | Experimental CPU INT8 eligible linears, or MPS FP16 decoder output projection. Disk artifacts are shared across profiles. |

## Measured on Apple M2

| Backend | Quality median | Balanced median | Transformer speedup | Exact RVQ speedup | Full generation speedup |
|---|---:|---:|---:|---:|---:|
| CPU | 1,586 ms | 286 ms | 5.58× | 1.38× | 5.54× |
| MPS | 762 ms | 137 ms | 4.89× | 1.59× | 5.58× |

Balanced matched the reference concept and text-token arrays on all 12 runs per backend across three authored prompts. Process peak RSS was 1.51 GiB on CPU and 0.97 GiB on MPS. MPS driver allocation was 1.19 GiB at the final sample, reported separately because the M2 uses unified memory. [Measurement protocol and full results](docs/performance.md).

```bash
ncm-benchmark --device cpu --profile quality --output results/quality.json
ncm-benchmark --device cpu --profile balanced --output results/balanced.json
python -m benchmarks.compare results/quality.json results/balanced.json
python -m benchmarks.quality results/balanced.json
```

Replace `cpu` with `mps` for Apple GPU timing. Run each profile in a fresh process. [Optimization experiments](docs/optimization-research.md).

## Adapt for dialogue

Training data is JSONL. Each line contains an array of chronological objects with `user` and `assistant` strings. Each turn supervises all target concept scales using the frozen neural tokenizer and teacher-forced residual latents. The default trains the generator heads. `--full` trains the whole generator.

```bash
HF_HUB_OFFLINE=1 ncm-train --train training/smoke.jsonl \
  --output artifacts/badge-dialogue --device cpu --steps 240 --learning-rate 0.002
HF_HUB_OFFLINE=1 ncm-chat
```

Use a fresh output directory. Choose the resulting model in the interface. The smoke corpus contains two two-turn exchanges. The trained model recalled both seen colors but failed to recall an unseen color. [Dialogue validation](docs/dialogue.md).

## Development

```bash
python -m pip install '.[dev]'
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 python -m pytest -q
HF_HUB_OFFLINE=1 NCM_TEST_ARTIFACTS=artifacts python -m pytest -q -m model
```

See the [architecture](docs/architecture.md), [research and license audit](docs/research-audit.md), and [roadmap](docs/optimization-roadmap.md). The implementation supports the official XSum architecture and dialogue checkpoints derived from it. Model outputs are limited to the checkpoint's fixed 48-position neural decoder. The app serializes inference to bound model residency. MLX is an isolated research experiment.

MIT. The [license](LICENSE) preserves the copyright notice for the [Hazy Research reference code](https://github.com/HazyResearch/ncm).
