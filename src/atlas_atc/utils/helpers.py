"""General utilities for ATLAS."""

import logging
import random
import time
from collections.abc import Iterator
from typing import Any, TypeVar

import numpy as np
import torch


def setup_logger(name: str = "atlas", level: str = "INFO") -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # No manual seed for MPS, but included for completeness
        pass

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, desc: str = "Elapsed") -> None:
        self.desc = desc
    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = time.time() - self.start
        print(f"{self.desc}: {elapsed:.3f}s")

T = TypeVar('T')

def batch_iterable(iterable: Any, batch_size: int) -> Iterator[Any]:
    """Yield successive batches from an iterable."""
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]

