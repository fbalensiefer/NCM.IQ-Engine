# NCM Chat

NCM Chat is a Rust inference engine, local web app, and dialogue adaptation tool for Next-Concept Modeling. The Transformer predicts a full scale of concept codes in parallel, and a multiscale residual VQ-VAE decodes the accumulated latent into text. The executable loads the official XSum checkpoint directly. [NCM paper](https://arxiv.org/pdf/2610.14162v1)

## Quick start

Install Rust 1.99 or newer, a C++20 toolchain, `curl`, `unzip`, and `shasum`. The Apple Silicon SDK setup also uses `jq`. The setup script downloads and verifies the native LibTorch 2.13 SDK for Apple Silicon macOS and Linux on x86-64 or ARM64.

```sh
LIBTORCH=$(bash scripts/libtorch.sh)
export LIBTORCH
cargo build --release --locked

target/release/ncm-chat download
target/release/ncm-chat serve
```

Open [the local interface](http://127.0.0.1:8000). The download command verifies the pinned checkpoint and vocabulary files and reuses verified local copies. The official checkpoint pair occupies 624 MiB. Once those files are available, inference, serving, benchmarks, and dialogue adaptation run offline.

The server binds to loopback and saves sessions in `sessions/chat.sqlite3`. Use `serve --help` to choose another model directory, vocabulary, session database, or port. [Local HTTP API](docs/api.md)

## Models and backends

The official XSum checkpoint summarizes the current passage. Dialogue-adapted checkpoints can use earlier turns as context. The current engine supports XSum and dialogue checkpoints derived from its architecture.

CPU and Apple MPS were verified on an M2. Automatic selection chooses an available accelerator and falls back to CPU. CUDA requires a compatible LibTorch SDK and remains pending hardware validation.

| Profile | Behavior |
|---|---|
| Quality | FP32, full padded-prefix evaluation at every scale. |
| Balanced | FP32, trimmed prefix, reusable prefix K/V, and cached exact RVQ norms. |
| Ultra-light | Experimental CPU dynamic INT8 linears or MPS FP16 decoder projection. |

```sh
target/release/ncm-chat generate --backend cpu "The library opens its reading room on Monday."
target/release/ncm-chat reconstruct --backend mps "Exact RVQ reconstruction runs locally."
```

## Benchmarks and tests

Balanced inference measured a median of **259 ms on CPU** and **133 ms on MPS** on the Apple M2. It matched Quality's concept and text-token arrays on every measured generation and reconstruction request. [Methodology and raw results](docs/performance.md)

```sh
target/release/ncm-chat benchmark --backend cpu --profile quality --output results/quality.json
target/release/ncm-chat benchmark --backend cpu --profile balanced --output results/balanced.json
target/release/ncm-chat compare results/quality.json results/balanced.json

cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --release --locked --test official -- --ignored
```

The last command requires the official checkpoints in `artifacts/` or the directory set by `NCM_TEST_ARTIFACTS`. CI runs lint and unit tests. The manual benchmark workflow adds the official checkpoint test and runs each profile in a fresh process.

## Adapt for dialogue

```sh
target/release/ncm-chat train --backend cpu --train training/smoke.jsonl \
  --output artifacts/my-dialogue --steps 240 --learning-rate 0.002
target/release/ncm-chat serve
```

Use a fresh output directory and select the resulting model in the web interface. The smoke corpus verifies the training and dialogue pipeline. General dialogue quality requires a separate held-out evaluation. [Data format, training options, and measured smoke results](docs/dialogue.md)

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Rust inference engine, Axum API, SQLite sessions, trainer, and benchmark CLI. Small C++ bindings expose LibTorch MPS telemetry and packed INT8 linears. |
| `frontend/` | Browser assets embedded in the executable. |
| `benchmarks/`, `training/` | Benchmark prompts and the dialogue smoke corpus. |
| `tests/`, `.github/workflows/` | Official-checkpoint regression test, CI, and benchmark automation. |
| `docs/` | [Architecture](docs/architecture.md), [research and licensing](docs/research-audit.md), [optimization notes](docs/optimization-research.md), and [roadmap](docs/optimization-roadmap.md). |

MIT. The [license](LICENSE) preserves the upstream Hazy Research notice.
