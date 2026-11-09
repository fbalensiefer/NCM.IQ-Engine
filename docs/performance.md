# Python reference performance

Apple M2 with 8 GB unified memory, macOS 26.6.2, Python 3.12.16, PyTorch 2.15.0, four CPU threads and batch size one. The [raw JSON reports](results/) record three authored prompts, two warm-up requests, 12 timed requests and seed 2026. P95 uses nearest rank. Each profile ran in a separate process. The first request includes allocator and kernel warm-up. Load timing starts after the Python runtime and its imports have initialized.

| Model and profile | Device | Median ms | P95 ms | First generation ms | Load ms | Peak process RSS GiB | Requests/s |
|---|---|---:|---:|---:|---:|---:|---:|
| NCM Quality | CPU | 1,586 | 1,620 | 1,642 | 1,192 | 1.37 | 0.63 |
| NCM Balanced | CPU | 286 | 343 | 305 | 1,193 | 1.51 | 3.37 |
| NCM Ultra-light | CPU | 237 | 302 | 349 | 1,238 | 1.70 | 3.98 |
| NCM Quality | MPS | 762 | 778 | 1,006 | 1,780 | 1.33 | 1.31 |
| NCM Balanced | MPS | 137 | 194 | 283 | 2,375 | 0.97 | 6.91 |
| NCM Ultra-light | MPS | 139 | 158 | 271 | 2,069 | 0.97 | 7.02 |
| T5-small, greedy | CPU | 187 | 207 | 532 | 251 | 0.59 | 5.54 |

The official NCM checkpoint pair is 653,981,290 bytes. T5-small is 242,043,056 bytes. MPS snapshots at the end of profiling for Quality, Balanced and Ultra-light were 0.66, 0.61 and 0.58 GiB of live tensors and 1.07, 1.19 and 1.19 GiB of driver allocations, respectively. RSS and driver bytes use different accounting and may overlap on unified memory. The MPS API provides allocation snapshots. System-wide unified-memory peak and energy were unmeasured.

## Component timings

Separate synchronized timing passes supplied the stage medians. The application uses these timers to expose preprocessing, input embedding and transfer, per-level attention stacks, sampling, RVQ, neural decoder, text selection and output handling. The XSum decoder's CPU FP64 text-selection timer includes transfer of the logits. Synchronization wait is included at detailed stage boundaries. Stage medians need not add to the steady median. The Transformer total includes one prefix stack in Balanced and includes the target projection and head in Balanced. Quality instrumentation starts at the first Transformer block and stops after final normalization.

| Component | CPU Quality ms | CPU Balanced ms | Speedup | MPS Quality ms | MPS Balanced ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Full generation | 1,586.20 | 286.37 | 5.54× | 762.25 | 136.64 | 5.58× |
| Transformer | 1,540.65 | 276.33 | 5.58× | 730.91 | 149.57 | 4.89× |
| Neural decoder forward | 5.40 | 5.62 | 0.96× | 4.58 | 4.60 | 1.00× |
| Decoder token selection | 10.08 | 4.78 | 2.11× | 10.79 | 5.57 | 1.94× |
| Neural encoder, reconstruction | 0.13 | 0.13 | 0.96× | 0.30 | 0.32 | 0.94× |
| Exact RVQ search, reconstruction | 10.19 | 7.40 | 1.38× | 10.85 | 6.81 | 1.59× |
| Full neural reconstruction | 18.81 | 16.26 | 1.16× | 20.11 | 17.27 | 1.16× |

The upstream CUDA-oriented model was initially run on CPU with its input, sampling, reconstruction and decoder logic intact. Its warmed median was 1,626 ms on the first prompt. Its unmodified area-pooling path failed on MPS. The Quality baseline uses the same weights and model math with a device-compatible pooling path and CPU FP64 decoder penalties. Balanced's reduction is prompt-length dependent. Reuse still avoids recomputing the prefix for longer input.

## Quality protocol

On the measured prompt set, Balanced matched Quality for all 12 generation and 12 reconstruction concept and token arrays per backend. The same matched on MPS Ultra-light. CPU Ultra-light matched none of the 12 generation arrays and 4 of 12 reconstructed token arrays. These 12 runs repeat three distinct inputs.

The separate lexical evaluation uses one authored reference per distinct prompt. Mean ROUGE-L F1 was 0.100 for NCM Quality and Balanced, 0.081 for CPU Ultra-light and 0.219 for T5-small. The three passages are a smoke set. The reports include raw outputs so fidelity and hallucinated details can be inspected. The T5 comparison uses the same three prompts, batch size, CPU threads and 512 source and 48 output token caps. Vocabularies, EOS behavior, parameter counts and training tasks differ. Dialogue evaluation is separate. [Dialogue smoke test](dialogue.md).

To reproduce the optional T5 comparison, download only the needed files once and use the local path afterward.

```bash
hf download google-t5/t5-small config.json tokenizer.json tokenizer_config.json \
  generation_config.json model.safetensors \
  --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --local-dir artifacts/t5-small
HF_HUB_OFFLINE=1 python -m benchmarks.run_t5 --model artifacts/t5-small --output results/t5.json
python -m benchmarks.quality results/t5.json
```
