# Optimization research

## Transformer

The reference generator recomputes prefix hidden states for each hierarchy level. Its mask prevents prefix queries from seeing target positions, and RoPE gives the same prefix positions at every level. Balanced computes per-layer prefix keys and values once, removes right padding, and runs the shorter target blocks through PyTorch SDPA with the appropriate rectangular masks. The coarse query follows the same path as finer queries. The cache is cleared after each request. No state is reused across unrelated conversations. The measured component and end-to-end results are in [performance.md](performance.md).

The released model already uses SDPA. The path retains noncausal attention inside a level, with a broadcastable prefix-validity mask for classifier-free guidance and reserved padding tokens. FlashAttention and memory-efficient SDPA can be selected by the PyTorch dispatcher where the device, mask, dtype and shape permit. The application leaves kernel choice to PyTorch. Causal language-model kernels and speculative token decoding address different sequence dependencies. [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [FlashAttention paper](https://arxiv.org/abs/2205.14135).

A fixed-shape CPU decoder experiment on the official weights compared eager and Inductor. The host retry measured 5.27 ms eager and 5.64 ms compiled median, with 8.72 s to compile and execute the first call. Maximum absolute logit error was 0.000044 and greedy IDs matched on the test latent. The initial sandbox run also showed a Homebrew cache permission error during toolchain discovery. Compilation stays out of runtime. Reproduce with `python -m benchmarks.compile_decoder`. A whole-generator compilation benchmark remains open because the generator traverses distinct scale shapes and cache states. [PyTorch compiler documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html).

## Neural tokenizer and decoder

The XSum VQ-VAE uses a 384-wide latent and nine 4,096-entry codebooks. Its default source path uses a raw embedding. Reconstruction tests exercise the full learned encoder and RVQ. The reference distance calculation already uses a matrix multiplication, so the optimized search precomputes frozen codebook squared norms, preserves FP32 distance arithmetic, and skips partial reconstructions and the terminal residual update when only IDs are requested. Codebooks stay on-device. This improved exact lookup by 1.38× on CPU and 1.59× on MPS, with matching reconstruction IDs for every sample tested. The source embedding and neural encoder are small fractions of latency. Kernel-level bandwidth and energy counters were unavailable.

The MPS pooling fallback builds the same adaptive-bin boundaries, gathers the corresponding on-device elements, and reduces over the valid entries. It supports checkpoint scale lengths for which PyTorch MPS adaptive pooling raises an error. CPU/MPS parity is covered for a non-divisible shape, and the real reconstruction outputs matched across profiles. No approximate search or custom native operator was introduced. [PyTorch MPS backend](https://docs.pytorch.org/docs/main/notes/mps.html).

Generation reuses its accumulated latent for neural decoding and applies the last scale once. The reference repetition policy performs FP64 scans over the vocabulary at every position. For a single request, Balanced broadcasts the zero-count frequency adjustment, then updates only previously selected tokens before the argmax. Neural decoder outputs still cover all token positions in parallel. The sparse policy improved selection by 2.11× on CPU and 1.94× on MPS in this workload.

On MPS, an isolated FP16 output projection test ran 24 synchronized decoder forwards across three fixed latents after warm-up. It moved median decoder time from 4.89 to 4.36 ms and matched all 24 selected token arrays. The integrated MPS Ultra-light profile preserved all measured arrays and used about 37 MiB fewer live MPS tensor bytes than Balanced. End-to-end median was slightly slower at 139 ms versus 137 ms. BF16 and full-stack mixed precision remain unmeasured.

## Quantization and memory

CPU Ultra-light uses the backend's dynamic INT8 linear kernels for eligible generator and decoder projections. Codebooks and latent math remain FP32. It improved CPU median end-to-end latency from 286 to 237 ms versus Balanced, changed generated tokens on all measured prompts, slowed the decoder forward, and increased process peak RSS during load and execution. The profile remains experimental. QNNPACK reported a reduce-range behavior difference, and PyTorch emits deprecation warnings for the eager quantization API. TorchAO grouped INT4, per-channel alternatives and quantization-aware training remain candidates for a separately validated weight format. [PyTorch quantize_dynamic](https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_dynamic.html), [TorchAO inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html).

Balanced memory-maps released weights and drops EMA-only training buffers. On MPS it reduced measured process peak RSS, while the ending driver allocation increased. On CPU it increased measured peak RSS. Cold-load mappings, retained tensor storage, and Metal's allocation cache are accounted for separately in the reports. The app lazily loads one profile per worker and keeps its accelerator tensors resident between requests. Request batching, persistent staging buffers and native Metal kernels would require broader shape-specific measurement.

## MLX and native acceleration

MLX 0.32.4 and PyTorch 2.15.0 ran equivalent masked SDPA kernels on the M2, using batch one, 12 heads, 64 channels per head and 48 target queries. The input arrays were synthetic, preallocated and resident for the measured calls. Each median used 64 synchronized calls after eight warm-ups.

| Dtype and prefix slots | PyTorch MPS ms | MLX ms | Maximum absolute difference |
|---|---:|---:|---:|
| FP32, 32 | 0.281 | 0.271 | 0 |
| FP32, 512 | 0.375 | 0.341 | 0 |
| FP16, 32 | 0.329 | 0.279 | 0.000061 |
| FP16, 512 | 0.384 | 0.317 | 0.000122 |

The isolated differences are tens of microseconds per call and omit the generator's projections, cache, quantizer and decoder. The experiment does not establish an end-to-end advantage for an MLX port. Reproduce by installing MLX separately and running `python -m benchmarks.attention_mlx`. [MLX attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html), [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

Exact RVQ and decoder projections already use platform matrix kernels. A C++, SIMD, or Metal implementation would need to beat those kernels or eliminate enough launch and allocation overhead. Distillation, codebook dimensionality changes, structured pruning and low-rank weights change the learned latent interface or weight approximation. Their acceptance gates are reconstruction parity or quality, held-out generation quality, and real memory and latency reductions.

A CPU run of `python -m benchmarks.compile_decoder --transformer-block` used the first official XSum block with a 512-slot padded source and four coarse target queries. The eager median was 16.13 ms and the compiled median 13.23 ms, with 6.49 s to compile and execute the first call. The output maximum absolute difference was 0.0000097. This one-block gain leaves per-scale compilation cost and end-to-end behavior open.
