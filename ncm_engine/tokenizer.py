from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_vocabulary(directory: Path, config):
    if not directory.is_dir():
        raise FileNotFoundError(f"Local text vocabulary is missing: {directory}")
    tokens = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    special = config.get("special_tokens", {})
    if special:
        tokens.add_special_tokens(dict(special))
    if tokens.pad_token_id is None:
        tokens.pad_token = tokens.eos_token
    return tokens


def prefix_ids(vocab, text: str, limit: int, width: int, keep_recent=False):
    ids = vocab.encode(text, add_special_tokens=False, verbose=False)
    ids = (ids[-limit:] if keep_recent else ids[:limit]) or [vocab.eos_token_id]
    return torch.tensor([ids + [vocab.pad_token_id] * (width - len(ids))]), len(ids)


def target_ids(vocab, text, width):
    ids = vocab.encode(text, add_special_tokens=False, verbose=False)[: width - 1]
    ids.append(vocab.eos_token_id)
    return ids + [vocab.pad_token_id] * (width - len(ids))
