import torch
from torch.nn import functional as F


def interpolate(x, *, size, mode):
    """Match adaptive average pooling bins on MPS for non-divisible lengths."""
    if mode != "area" or x.device.type != "mps" or x.shape[-1] % size[0] == 0:
        return F.interpolate(x, size=size, mode=mode)
    length, target = x.shape[-1], size[0]
    if length == target:
        return x
    start = torch.arange(target, device=x.device) * length // target
    stop = (
        torch.arange(1, target + 1, device=x.device) * length + target - 1
    ) // target
    # Every adaptive pooling bin is at most ceil(length / target) + 1 wide.
    offsets = torch.arange((length + target - 1) // target + 1, device=x.device)
    positions = start[:, None] + offsets
    valid = positions < stop[:, None]
    values = x.index_select(-1, positions.clamp_max(length - 1).flatten())
    values = values.unflatten(-1, positions.shape).masked_fill(~valid, 0)
    return values.sum(-1) / (stop - start).to(x.dtype)


class FrozenRVQ:
    """Exact nearest-code search with fixed codebook norms and no partial outputs."""

    def __init__(self, vq):
        self.vq = vq
        self.norms = [(table.weight.square()).sum(dim=1) for table in vq.embeddings]

    def encode(self, z):
        batch, length, width = z.shape
        residual = z.float().clone()
        codes = []
        for i, (level, table, norm) in enumerate(
            zip(self.vq.levels, self.vq.embeddings, self.norms)
        ):
            last = i == len(self.norms) - 1
            resized = residual if last else self.vq._down_for_quant(residual, i, level)
            flat = resized.reshape(-1, width)
            # Keep the original operation order so close codebook distances
            # are rounded the same way on the same backend.
            distances = (
                flat.square().sum(1, keepdim=True) + norm - 2 * (flat @ table.weight.T)
            )
            indices = distances.argmin(dim=1).view(batch, level)
            codes.append(indices)
            if not last:
                contribution = self.vq._up_to_full(table(indices), i, length)
                residual.sub_(self.vq._apply_quant_resi(contribution, i))
        return torch.cat(codes, dim=1)
