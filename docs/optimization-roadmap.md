# Optimization roadmap

| Work item | Acceptance criterion |
|---|---|
| Representative dialogue adaptation | Use a documented corpus with usage rights, a held-out evaluation split, and multi-turn factual and coherence checks. Publish checkpoint provenance and failures. |
| Representative XSum evaluation | Compare Quality and Balanced on a fixed licensed evaluation sample, report ROUGE and inspected factual consistency, and analyze near-tie disagreements. |
| Train a smaller NCM | Train a reduced-width VQ-VAE and generator against reconstruction, task quality and memory budgets. Compare codebook utilization and latency. |
| Persistent compressed weights | Evaluate TorchAO grouped INT4 and per-channel INT8 separately for Transformer and decoder. Compare file bytes, peak load memory, quality and cold start. |
| MLX feasibility | Port the tokenizer and generator together if broader kernel and trace measurements suggest an end-to-end win. Compare per-level logits, RVQ codes and decoded IDs with the FP32 reference. |
| Production profiling | Add shape-controlled long-input tests, isolated energy measurements and CUDA validation. Measure host RAM and GPU memory with platform-appropriate counters. |
| Request batching | Batch independent requests by model, profile and valid prefix width. Measure both queue latency and throughput and preserve session ordering. |
| Performance CI | Establish a multi-run baseline on dedicated hardware before enabling a median/P95 regression gate. Hosted CI currently checks correctness. |
