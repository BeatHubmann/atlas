"""General utilities for ATLAS."""

import logging
import time
import numpy as np
import random
import torch

def setup_logger(name="atlas", level="INFO"):
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def set_seed(seed: int = 42):
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
    def __init__(self, desc="Elapsed"):
        self.desc = desc
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.desc}: {elapsed:.3f}s")

def batch_iterable(iterable, batch_size):
    """Yield successive batches from an iterable."""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

