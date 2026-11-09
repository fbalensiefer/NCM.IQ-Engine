# Performance and validation

The Rust release binary was tested on an Apple M2 with 8 GB unified memory, macOS 26.6.2, Rust 1.99.1, and LibTorch 2.13.0. Each profile used four CPU threads, batch size one, two warm-up requests, and 12 timed requests drawn from three authored prompts. Profiles ran in separate processes. P95 uses nearest rank. Stage timings come from a separate synchronized pass. [Prompts](../benchmarks/prompts.json) · [Raw reports](results/rust/)

| Backend and profile | Median ms | P95 ms | First generation ms | Model load ms | Peak process RSS GiB |
|---|---:|---:|---:|---:|---:|
| CPU Quality | 1,594 | 1,779 | 1,614 | 237 | 0.90 |
| CPU Balanced | 259 | 296 | 260 | 239 | 0.94 |
| CPU Ultra-light | 211 | 286 | 319 | 283 | 1.11 |
| MPS Quality | 835 | 845 | 1,231 | 535 | 0.44 |
| MPS Balanced | 133 | 168 | 261 | 687 | 0.51 |
| MPS Ultra-light | 131 | 167 | 212 | 648 | 0.55 |

MPS Balanced reported 1.07 GiB of driver allocation at the final sample. Process RSS and driver allocation use separate accounting and may overlap on unified memory. System-wide peak memory and energy were unmeasured.

| Balanced versus Quality | CPU | MPS |
|---|---:|---:|
| Full generation speedup | 6.15× | 6.30× |
| Transformer speedup | 6.34× | 5.41× |
| Exact RVQ search speedup | 1.36× | 1.51× |
| Full neural reconstruction speedup | 1.16× | 1.34× |

Balanced matched Quality's concept and text-token arrays on all 12 generation and 12 reconstruction requests per backend. The official checkpoint regression test uses the recorded Quality outputs for exact comparison on Apple Silicon. Its checks on other hosts cover output dimensions, decoded text, and timing.

MPS Ultra-light matched the measured arrays and reduced median decoder forward time from 4.50 to 3.91 ms versus Balanced. CPU Ultra-light changed every measured generation array and matched four of 12 reconstructed token arrays. Its QNNPACK kernel also emitted a reduce-range warning. The workload covers a small set of prompts. Dialogue evaluation is separate. [Dialogue smoke results](dialogue.md)

## Reproduce

Run from the repository root after building the release binary and downloading the official artifacts. Use `cpu` or `mps` as the backend. The comparison command checks common benchmark settings and reports stage speedups and output-hash parity.

```sh
backend=cpu
for profile in quality balanced ultra; do
  target/release/ncm-iq benchmark --backend "$backend" --profile "$profile" \
    --output "results/$backend-$profile.json"
done
target/release/ncm-iq compare "results/$backend-quality.json" "results/$backend-balanced.json"
```
