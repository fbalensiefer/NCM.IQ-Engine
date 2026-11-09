from pathlib import Path
from threading import Lock

import torch
from omegaconf import OmegaConf

from backend.config import Model
from ncm_engine.device import select_device
from ncm_engine.inference import Engine
from ncm_engine.sampling import Sampling


class ModelManager:
    def __init__(self, root, vocabulary):
        self.models = {}
        for path in sorted(Path(root).glob("*/config.yaml")):
            config = OmegaConf.load(path)
            if config.get("task") in {"xsum", "dialogue"} and all(
                (path.parent / config.checkpoints[key]).is_file()
                for key in ("tokenizer", "generator")
            ):
                name = path.parent.name
                self.models[name] = Model(
                    name,
                    path.parent,
                    Path(vocabulary),
                    config.task,
                    int(config.model.prefix_len),
                )
        self.lock = Lock()
        self.engine = None
        self.key = None

    def generate(self, name, backend, profile, text, sampling, context_limit):
        device = select_device(backend)
        if profile == "ultra" and backend == "auto" and device.type == "cuda":
            device = select_device("cpu")
        key = (name, device.type, profile)
        model = self.models[name]
        with self.lock:
            if key != self.key:
                old_device = self.engine.device.type if self.engine else None
                self.engine = self.key = None
                if old_device == "mps":
                    torch.mps.empty_cache()
                elif old_device == "cuda":
                    torch.cuda.empty_cache()
                self.engine = Engine(
                    model.directory, model.vocabulary, device.type, profile
                )
                self.key = key
            settings = Sampling(**{**vars(self.engine.defaults), **sampling})
            result = self.engine.generate(
                text, sampling=settings, context_limit=context_limit, detailed=True
            )
            stats = {
                "backend": device.type,
                "profile": profile,
                "model_load_ms": self.engine.load_ms,
                "checkpoint_bytes": self.engine.checkpoint_bytes,
                "rss_bytes": result.rss_bytes,
                "milliseconds": result.milliseconds,
            }
            if device.type == "mps":
                stats["accelerator_allocated_bytes"] = (
                    torch.mps.current_allocated_memory()
                )
            elif device.type == "cuda":
                stats["accelerator_allocated_bytes"] = torch.cuda.memory_allocated()
            return result.text, stats
