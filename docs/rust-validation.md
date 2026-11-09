# Rust validation

The Rust release binary was tested on an Apple M2 with 8 GB unified memory, macOS 26.6.2, Rust 1.99.1 and LibTorch 2.13.0. Each profile used four CPU threads, batch size one, two warm-ups and 12 timed requests drawn from three authored prompts. Benchmarks ran in separate processes. P95 uses nearest rank. The decoder and Transformer stage medians come from a separate synchronized timing pass. [Raw reports](results/rust/).

| Backend and profile | Median ms | P95 ms | First generation ms | Model load ms | Peak process RSS GiB |
|---|---:|---:|---:|---:|---:|
| CPU Quality | 1,594 | 1,779 | 1,614 | 237 | 0.90 |
| CPU Balanced | 259 | 296 | 260 | 239 | 0.94 |
| CPU Ultra-light | 211 | 286 | 319 | 283 | 1.11 |
| MPS Quality | 835 | 845 | 1,231 | 535 | 0.44 |
| MPS Balanced | 133 | 168 | 261 | 687 | 0.51 |
| MPS Ultra-light | 131 | 167 | 212 | 648 | 0.55 |

MPS Balanced reported 1.07 GiB driver allocation at the final sample. Process RSS and driver allocation use separate accounting and may overlap. MPS system-wide unified-memory peak and energy were unmeasured.

| Balanced versus Quality | CPU | MPS |
|---|---:|---:|
| Full generation speedup | 6.15× | 6.30× |
| Transformer speedup | 6.34× | 5.41× |
| Exact RVQ search speedup | 1.36× | 1.51× |
| Full neural reconstruction speedup | 1.16× | 1.34× |

Rust Balanced matched the concept and text-token arrays from the saved Python reference on all 12 generation and 12 reconstruction requests for each backend. It also matched Rust Quality on these requests. The earlier Python Balanced runs measured 286 ms CPU and 137 ms MPS. These comparisons cover the complete builds, including LibTorch version changes. They do not isolate the effect of the host language. [Python baseline and conventional T5 comparator](performance.md).

MPS Ultra-light preserved the measured arrays and reduced median neural decoder forward time to 3.91 ms from 4.50 ms in Rust Balanced. CPU Ultra-light changed all generated concept and token arrays and matched four of 12 reconstructed token arrays. CPU INT8 uses LibTorch's packed dynamic linear kernels, and the M2's QNNPACK build emitted its reduce-range warning.

## Dialogue and local API

The Rust trainer adapted the generator heads on the four smoke-corpus examples for 240 steps at learning rate 0.002 and seed 2026. The reported concept loss went from 2.4709 on the first step to 0.4158 on the final step. The web server loaded the native trainer's Safetensors output. It recalled teal and purple after a follow-up question, repeated a word in both replies, and answered purple for the unseen color orange. [Raw HTTP replies](results/rust/dialogue-smoke.jsonl).

The [loopback smoke script](../scripts/smoke-http.sh) verifies embedded assets, two-turn generation, retrieval of persisted state, stale revision rejection, reset, and XSum inference. To reproduce with a fresh model directory:

```sh
target/release/ncm-chat train --backend cpu --train training/smoke.jsonl \
  --output artifacts/my-dialogue --steps 240 --learning-rate 0.002
bash scripts/smoke-http.sh target/release/ncm-chat my-dialogue
```

The three Rust unit tests and the optional official-checkpoint test passed. The official test exercised generation and reconstruction on CPU and MPS. Linux CI and CUDA hardware validation remain pending. MLX experiments remain reference research.
