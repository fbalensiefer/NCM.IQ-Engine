from pathlib import Path

import torch


def read_checkpoint(path: Path):
    """Read the released ZIP checkpoints without copying storage into the heap."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
