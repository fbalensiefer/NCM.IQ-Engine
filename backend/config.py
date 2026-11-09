from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    name: str
    directory: Path
    vocabulary: Path
    task: str
    context_limit: int
