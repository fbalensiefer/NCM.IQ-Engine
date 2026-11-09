from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.0
    cfg_scale: float = 0.0

    def validate(self, vocab_sizes):
        if (
            not isfinite(self.temperature)
            or self.temperature < 0
            or not 0 <= self.top_k <= min(vocab_sizes)
            or not 0 <= self.top_p <= 1
            or not isfinite(self.cfg_scale)
            or self.cfg_scale < 0
        ):
            raise ValueError("Invalid concept sampling parameters")
