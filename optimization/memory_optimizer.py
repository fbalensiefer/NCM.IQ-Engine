from pathlib import Path

import torch


def read_checkpoint(path: Path):
    """Read the released ZIP checkpoints without copying storage into the heap."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def release_training_buffers(vq):
    """Discard statistics used only by the VQ-VAE's training forward."""
    for i in range(len(vq.embeddings)):
        for name in ("ema_cluster_size", "ema_w", "codebook_hit"):
            setattr(vq, f"{name}_{i}", None)
