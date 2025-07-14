"""Linear extrapolation baseline model."""

import numpy as np
import time
from typing import Dict, Any
from .base import TrajectoryPredictor, PredictionResult


class LinearExtrapolationPredictor(TrajectoryPredictor):
    """Simple linear extrapolation using velocity vector."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("Linear Extrapolation", device)
        self.is_trained = True  # No training required
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Linear extrapolation requires no training."""
        pass
    
    def predict(self, X: np.ndarray, horizon_minutes: int) -> PredictionResult:
        """
        Predict future positions using linear extrapolation.
        
        Args:
            X: Historical trajectory points [batch_size, seq_len, features]
               Features: [lat, lon, alt, timestamp]
            horizon_minutes: Prediction horizon in minutes
        
        Returns:
            PredictionResult with predicted positions
        """
        start_time = time.time()
        
        # Use last two points to calculate velocity
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 historical points for linear extrapolation")
        
        # Extract last two positions
        last_pos = X[:, -1, :3]  # [lat, lon, alt]
        prev_pos = X[:, -2, :3]
        
        # Calculate time difference
        last_time = X[:, -1, 3]
        prev_time = X[:, -2, 3]
        dt = last_time - prev_time
        
        # Calculate velocity vector
        velocity = (last_pos - prev_pos) / dt[:, np.newaxis]
        
        # Generate predictions
        prediction_times = np.arange(1, horizon_minutes + 1) * 60  # Convert to seconds
        predictions = []
        
        for t in prediction_times:
            pred_pos = last_pos + velocity * t
            predictions.append(pred_pos)
        
        predictions = np.stack(predictions, axis=1)  # [batch_size, horizon, 3]
        
        computation_time = time.time() - start_time
        
        return PredictionResult(
            predictions=predictions,
            confidence=0.5,  # Fixed confidence for linear model
            computation_time=computation_time,
            metadata={
                "model": self.name,
                "horizon_minutes": horizon_minutes,
                "velocity_magnitude": np.linalg.norm(velocity, axis=1).mean()
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "name": self.name,
            "type": "baseline",
            "parameters": 0,
            "training_required": False,
            "description": "Linear extrapolation using velocity vector from last two points"
        }

