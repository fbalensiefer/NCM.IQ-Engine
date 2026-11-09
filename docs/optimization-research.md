# Inference optimizations

## Transformer prefix reuse

The attention mask prevents source queries from seeing target positions. Balanced trims right padding and evaluates the source together with the first concept scale. It keeps only the source keys and values from each layer, then reuses them for the remaining rectangular attention passes. Source positions and their RoPE coordinates remain fixed across scales. Target positions attend bidirectionally within each scale and use the accumulated latent from earlier scales. LibTorch dispatches scaled dot-product attention for the selected backend.

The largest category in the synchronized block diagnostics was linear projections on both CPU and MPS. A preallocated KV buffer reduced end-to-end median time by about 1% in the tested runs and increased the final MPS driver allocation from 1,156 MB to 1,198 MB. The combined pass keeps LibTorch's concatenation for subsequent scales and reduced median steady generation latency by 9% on CPU and 11% on MPS in Balanced. CPU Ultra-light keeps separate passes because dynamic INT8 quantization chooses activation scales from each input. Gains vary with prefix length and backend. [Measurement protocol](performance.md)

## Exact RVQ and neural decoding

The neural tokenizer computes squared Euclidean distances to the released codebooks using matrix multiplication. Balanced caches codebook squared norms, keeps the tables on the inference device, and skips unused partial reconstructions when finding code IDs. On MPS, a device-local gather and reduction reproduces adaptive pooling bins for lengths that do not divide evenly.

Generation reuses the final cumulative latent for parallel neural decoding. Text selection applies the checkpoint's repetition policy to the computed logits. It tracks the previously selected tokens so it can update their penalties without recomputing token counts across the whole vocabulary.

Balanced exact RVQ search was 1.36× faster on CPU and 1.51× faster on MPS than Quality. All measured code and token arrays matched. [Raw stage timings](results/rust/)

## Precision and memory

The checkpoint reader skips the unused EMA and codebook hit-count tensors. The server keeps one model resident and replaces it on model, backend, or profile changes.

Ultra-light uses LibTorch packed dynamic INT8 linears on CPU and a FP16 decoder output projection on MPS. CPU INT8 reduced median generation from 259 to 211 ms versus Balanced, increased peak process RSS, and changed the generated arrays. On MPS, the FP16 projection reduced median decoder forward time from 4.50 to 3.91 ms. Output arrays matched on the measured prompt set. [Profile results](performance.md)
