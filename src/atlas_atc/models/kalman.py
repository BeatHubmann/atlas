"""Kalman filter trajectory prediction model."""

import time
from typing import Any

import numpy as np

from .base import PredictionResult, TrajectoryPredictor


class KalmanFilterPredictor(TrajectoryPredictor):
    """Kalman filter for trajectory prediction with constant velocity model."""

    def __init__(self, device: str = "cpu"):
        super().__init__("Kalman Filter", device)

        # State vector: [x, y, z, vx, vy, vz]
        self.state_dim = 6
        self.obs_dim = 3

        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.1

        # Observation noise covariance
        self.R = np.eye(self.obs_dim) * 1.0

        # Initial state covariance
        self.P_init = np.eye(self.state_dim) * 10.0

        self.is_trained = True  # No training required

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Kalman filter requires no training."""
        pass

    def _create_transition_matrix(self, dt: float) -> np.ndarray:
        """Create state transition matrix for constant velocity model."""
        F = np.eye(self.state_dim)
        F[0, 3] = dt  # x = x + vx * dt
        F[1, 4] = dt  # y = y + vy * dt
        F[2, 5] = dt  # z = z + vz * dt
        return F

    def _create_observation_matrix(self) -> np.ndarray:
        """Create observation matrix (observe position only)."""
        H = np.zeros((self.obs_dim, self.state_dim))
        H[0, 0] = 1  # observe x
        H[1, 1] = 1  # observe y
        H[2, 2] = 1  # observe z
        return H

    def _initialize_state(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Initialize state vector from first two observations."""
        if len(positions) < 2:
            raise ValueError("Need at least 2 points to initialize Kalman filter")

        # Initial position
        x0 = positions[0]

        # Initial velocity estimate
        dt = times[1] - times[0]
        v0 = (positions[1] - positions[0]) / dt

        # State vector: [x, y, z, vx, vy, vz]
        state = np.concatenate([x0, v0])
        return state

    def predict(self, X: np.ndarray, horizon_minutes: int) -> PredictionResult:
        """
        Predict future positions using Kalman filter.

        Args:
            X: Historical trajectory points [batch_size, seq_len, features]
               Features: [lat, lon, alt, timestamp]
            horizon_minutes: Prediction horizon in minutes

        Returns:
            PredictionResult with predicted positions
        """
        start_time = time.time()

        batch_size = X.shape[0]
        predictions = []

        H = self._create_observation_matrix()

        for i in range(batch_size):
            # Extract trajectory for this aircraft
            positions = X[i, :, :3]  # [seq_len, 3]
            times = X[i, :, 3]  # [seq_len]

            # Initialize state
            state = self._initialize_state(positions, times)
            P = self.P_init.copy()

            # Forward pass through historical data
            for t in range(1, len(positions)):
                dt = times[t] - times[t - 1]
                F = self._create_transition_matrix(dt)

                # Predict step
                state = F @ state
                P = F @ P @ F.T + self.Q

                # Update step
                z = positions[t]  # observation
                y = z - H @ state  # innovation
                S = H @ P @ H.T + self.R  # innovation covariance
                K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

                state = state + K @ y
                P = (np.eye(self.state_dim) - K @ H) @ P

            # Generate future predictions
            pred_positions = []

            for minute in range(1, horizon_minutes + 1):
                dt = minute * 60  # Convert to seconds
                F = self._create_transition_matrix(dt)

                # Predict future state
                future_state = F @ state
                pred_pos = future_state[:3]  # Extract position
                pred_positions.append(pred_pos)

            predictions.append(np.array(pred_positions))

        predictions = np.array(predictions)  # [batch_size, horizon, 3]

        computation_time = time.time() - start_time

        return PredictionResult(
            predictions=predictions,
            confidence=0.8,  # Higher confidence than linear extrapolation
            computation_time=computation_time,
            metadata={
                "model": self.name,
                "horizon_minutes": horizon_minutes,
                "state_dimension": self.state_dim,
                "process_noise": np.trace(self.Q),
                "observation_noise": np.trace(self.R),
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return model information."""
        return {
            "name": self.name,
            "type": "classical",
            "parameters": self.state_dim * (self.state_dim + self.obs_dim),
            "training_required": False,
            "description": "Kalman filter with constant velocity motion model",
        }
