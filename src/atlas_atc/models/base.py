"""Base classes for trajectory prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class PredictionResult:
    """Container for prediction results."""

    predictions: np.ndarray
    confidence: float
    computation_time: float
    metadata: dict[str, Any]


class TrajectoryPredictor(ABC):
    """Abstract base class for trajectory prediction models."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        self.is_trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on trajectory data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, horizon_minutes: int) -> PredictionResult:
        """Predict future trajectory points."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata and parameters."""
        pass

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        raise NotImplementedError("Subclasses must implement save_model")

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        raise NotImplementedError("Subclasses must implement load_model")


class TorchTrajectoryPredictor(TrajectoryPredictor):
    """Base class for PyTorch-based models."""

    def __init__(self, name: str, device: str = "mps"):
        super().__init__(name, device)
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion: torch.nn.Module | None = None

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.detach().cpu().numpy()

    def save_model(self, path: str) -> None:
        """Save PyTorch model state."""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                "model_info": self.get_model_info(),
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """Load PyTorch model state."""
        checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            raise ValueError("Model architecture not initialized")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.is_trained = True
