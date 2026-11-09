# Inference optimizations

## Transformer prefix reuse

The attention mask prevents source queries from seeing target positions. Source positions and their RoPE coordinates remain fixed across scales, so their layer-wise keys and values can be computed once per request. Balanced trims right padding and reuses that cache for each rectangular target attention pass. Target positions attend bidirectionally within each scale and use the accumulated latent from earlier scales. LibTorch dispatches scaled dot-product attention for the selected backend.

On the measured workload, the Balanced Transformer stage was 6.34× faster on CPU and 5.41× faster on MPS than Quality. Gains vary with prefix length and backend. [Measurement protocol](performance.md)

## Exact RVQ and neural decoding

The neural tokenizer computes squared Euclidean distances to the released codebooks using matrix multiplication. Balanced caches codebook squared norms, keeps the tables on the inference device, and skips unused partial reconstructions when finding code IDs. On MPS, a device-local gather and reduction reproduces adaptive pooling bins for lengths that do not divide evenly.

Generation reuses the final cumulative latent for parallel neural decoding. Text selection applies the checkpoint's repetition policy to the computed logits. It tracks the previously selected tokens so it can update their penalties without recomputing token counts across the whole vocabulary.

Balanced exact RVQ search was 1.36× faster on CPU and 1.51× faster on MPS than Quality. All measured code and token arrays matched. [Raw stage timings](results/rust/)

## Precision and memory

The checkpoint reader skips the unused EMA and codebook hit-count tensors. The server keeps one model resident and replaces it on model, backend, or profile changes.

Ultra-light uses LibTorch packed dynamic INT8 linears on CPU and a FP16 decoder output projection on MPS. CPU INT8 reduced median generation from 259 to 211 ms versus Balanced, increased peak process RSS, and changed the generated arrays. On MPS, the FP16 projection reduced median decoder forward time from 4.50 to 3.91 ms. Output arrays matched on the measured prompt set. [Profile results](performance.md)
