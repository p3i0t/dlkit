from typing import Optional
from pathlib import Path


class ModelDirectoryEmptyError(Exception):
    def __init__(self, dir: Optional[str | Path] = None) -> None:
        super().__init__()
        self.dir = dir

    def __str__(self) -> str:
        suffix = f" {self.dir}" if self.dir else ""
        return f"Model directory{suffix} is empty."
