import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic


def apply_cpu_int8(net, decoder):
    supported = torch.backends.quantized.supported_engines
    if torch.backends.quantized.engine not in supported:
        for backend in ("x86", "fbgemm", "qnnpack"):
            if backend in supported:
                torch.backends.quantized.engine = backend
                break
        else:
            raise ValueError("This PyTorch build has no supported INT8 CPU kernels")
    # Keep VQ, convolutions, normalization, and the prefix/level embeddings in FP32.
    # Dynamic INT8 converts eligible linear weights and quantizes activations at runtime.
    quantize_dynamic(net.transformer.blocks, {nn.Linear}, inplace=True)
    quantize_dynamic(net.lm_heads, {nn.Linear}, inplace=True)
    quantize_dynamic(decoder, {nn.Linear}, inplace=True)


class HalfProjection(nn.Module):
    """Keep the decoder stack in FP32 and store only its vocabulary head in FP16."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear.half()

    def forward(self, hidden):
        return self.linear(hidden.to(torch.float16))
