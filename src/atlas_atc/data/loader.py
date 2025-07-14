"""Data loader for SCAT dataset."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SCATDataLoader:
    """Loader for SCAT (System for Collection of ATC Data) dataset."""

    def __init__(self, data_path: Path | str):
        """
        Initialize the SCAT data loader.

        Args:
            data_path: Path to the SCAT data directory
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")

        # Find all JSON files
        self.trajectory_files = list(self.data_path.glob("**/*.json"))
        logger.info(f"Found {len(self.trajectory_files)} trajectory files")

    def load_single_trajectory(self, file_path: Path) -> dict[str, Any]:
        """Load a single trajectory from JSON file."""
        with open(file_path) as f:
            return json.load(f)

    def extract_trajectory_features(self, trajectory_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Extract features from a single trajectory.

        Args:
            trajectory_data: Raw trajectory data from JSON

        Returns:
            Tuple of (features array, metadata dict)
        """
        plots = trajectory_data.get('plots', [])
        if not plots:
            return np.array([]), {}

        # Extract position and time data
        positions = []
        times = []

        for plot in plots:
            # Extract lat/lon
            if 'I062/105' in plot:
                lat = plot['I062/105']['lat']
                lon = plot['I062/105']['lon']

                # Extract altitude (in flight levels, convert to feet)
                alt_fl = plot.get('I062/136', {}).get('measured_flight_level', 0)
                alt_ft = alt_fl * 100  # FL to feet

                # Extract time
                time_str = plot.get('time_of_track', '')

                positions.append([lat, lon, alt_ft])
                times.append(pd.to_datetime(time_str))

        if not positions:
            return np.array([]), {}

        positions_array = np.array(positions)
        times = pd.to_datetime(times)

        # Convert times to seconds from first observation
        time_deltas = (times - times[0]).total_seconds()

        # Combine position and time
        features = np.column_stack([positions_array, time_deltas])

        # Extract metadata
        fpl_base = trajectory_data.get('fpl', {}).get('fpl_base', [{}])[0]
        metadata = {
            'flight_id': trajectory_data.get('id'),
            'callsign': fpl_base.get('callsign'),
            'aircraft_type': fpl_base.get('aircraft_type'),
            'departure': fpl_base.get('adep'),
            'destination': fpl_base.get('ades'),
            'n_points': len(positions_array),
        }

        return features, metadata

    def load_trajectories(self, max_files: int | None = None) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """
        Load multiple trajectories from the dataset.

        Args:
            max_files: Maximum number of files to load (None for all)

        Returns:
            Tuple of (list of feature arrays, list of metadata dicts)
        """
        trajectories = []
        metadata_list = []

        files_to_load = self.trajectory_files[:max_files] if max_files else self.trajectory_files

        for i, file_path in enumerate(files_to_load):
            if i % 100 == 0:
                logger.info(f"Loading trajectory {i}/{len(files_to_load)}")

            try:
                data = self.load_single_trajectory(file_path)
                features, metadata = self.extract_trajectory_features(data)

                if len(features) > 10:  # Only keep trajectories with sufficient points
                    trajectories.append(features)
                    metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(trajectories)} valid trajectories")
        return trajectories, metadata_list

    def create_sequences(self, trajectories: list[np.ndarray],
                        sequence_length: int = 20,
                        prediction_horizons: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training from trajectories.

        Args:
            trajectories: List of trajectory arrays
            sequence_length: Length of input sequences
            prediction_horizons: List of prediction horizons in minutes

        Returns:
            Tuple of (sequences, targets) arrays
        """
        if prediction_horizons is None:
            prediction_horizons = [2, 5, 8]  # Default horizons in minutes

        X_list = []
        y_list = []

        for trajectory in trajectories:
            if len(trajectory) < sequence_length + max(prediction_horizons):
                continue

            # Create sequences from this trajectory
            for i in range(len(trajectory) - sequence_length - max(prediction_horizons)):
                # Input sequence
                X = trajectory[i:i + sequence_length]

                # Target positions at different horizons
                targets = []
                for horizon in prediction_horizons:
                    target_idx = i + sequence_length + horizon
                    if target_idx < len(trajectory):
                        targets.append(trajectory[target_idx, :3])  # lat, lon, alt
                    else:
                        targets.append(trajectory[-1, :3])  # Use last position

                X_list.append(X)
                y_list.append(np.array(targets))

        if not X_list:
            return np.array([]), np.array([])

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        return {
            "dataset": "SCAT",
            "version": "20161015_20161021",
            "n_files": len(self.trajectory_files),
            "data_path": str(self.data_path),
            "features": ["latitude", "longitude", "altitude_ft", "time_seconds"]
        }
