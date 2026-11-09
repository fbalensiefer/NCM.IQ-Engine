# NCM Chat

A Rust inference engine, local chat app, and dialogue adaptation tool for [Next-Concept Modeling](https://arxiv.org/pdf/2610.14162v1). The Transformer predicts one complete scale of concepts at a time. The multiscale residual VQ-VAE reconstructs the final latent and decodes it to text. The Rust executable loads the official XSum checkpoint archives directly and serves the web interface over loopback.

## Run locally

Install Rust 1.99 or newer, a C++ toolchain, `curl`, and `unzip`. The Apple Silicon SDK setup also uses `jq`. The setup script extracts the C++ headers and libraries for LibTorch 2.13 from the official PyTorch distribution. Build and inference require no Python interpreter.

```sh
LIBTORCH=$(bash scripts/libtorch.sh)
export LIBTORCH
cargo build --release --locked

target/release/ncm-chat download
target/release/ncm-chat serve
```

Open [the local interface](http://127.0.0.1:8000). Sessions are stored in `sessions/chat.sqlite3`. The download command checks the pinned artifact hashes and reuses verified local files. Inference, serving, benchmarks, and adaptation run offline. The official checkpoint pair is 624 MiB.

```sh
target/release/ncm-chat generate --backend cpu "The library opens its reading room on Monday."
target/release/ncm-chat reconstruct --backend mps "Exact RVQ reconstruction runs locally."
```

CPU and MPS were exercised on Apple M2. Auto selects an available accelerator and falls back to CPU. Linux CPU setup is in CI, and CUDA selection requires a CUDA-capable LibTorch build. The released XSum checkpoint summarizes the current passage. A dialogue-adapted checkpoint receives previous turns as context.

| Profile | Behavior |
|---|---|
| Quality | FP32, full padded-prefix evaluation at every scale. |
| Balanced | FP32, trimmed prefix, reusable prefix K/V, and cached exact RVQ norms. |
| Ultra-light | Experimental CPU dynamic INT8 linears or MPS FP16 decoder projection. |

## Benchmarks and tests

```sh
target/release/ncm-chat benchmark --backend cpu --profile quality --output results/quality.json
target/release/ncm-chat benchmark --backend cpu --profile balanced --output results/balanced.json
target/release/ncm-chat compare results/quality.json results/balanced.json

cargo test --locked
cargo test --release --locked --test official -- --ignored
```

The report separates end-to-end latency, per-scale Transformer time, RVQ, neural decoder time, and decoder token selection. It records code and token hashes for parity checks. See the [Rust validation report](docs/rust-validation.md), [prior reference results](docs/performance.md), and [optimization research](docs/optimization-research.md).

## Adapt for dialogue

Each line of the training file is a JSON array of chronological `user` and `assistant` pairs. The trainer freezes the VQ-VAE, encodes the target text, and supervises all target concept scales with teacher-forced residual latents. By default it adapts the generator heads. Use `--full` to train all generator weights.

```sh
target/release/ncm-chat train --backend cpu --train training/smoke.jsonl \
  --output artifacts/my-dialogue --steps 240 --learning-rate 0.002
target/release/ncm-chat serve
```

The output directory must be new. Select the resulting model in the web interface. The tiny [smoke corpus](training/smoke.jsonl) exercises the training and dialogue path. Held-out dialogue quality requires a separate evaluation.

## Repository layout

- `src/` contains the Rust inference engine, Axum API, SQLite sessions, adaptation, and benchmarks. Small C++ bindings expose MPS telemetry and LibTorch's packed INT8 linears.
- `frontend/` contains the web interface served from assets embedded in the binary.
- `docs/` contains the [architecture](docs/architecture.md), [checkpoint and license audit](docs/research-audit.md), and [research roadmap](docs/optimization-roadmap.md).
- The earlier Python implementation and its reports remain as the numerical and performance reference. Cargo builds and runs the application independently.

MIT. The [license](LICENSE) preserves the notice for [Hazy Research](https://github.com/HazyResearch/ncm).
