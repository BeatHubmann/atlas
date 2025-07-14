"""Benchmarking framework for trajectory prediction models."""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any
from pathlib import Path
import json
import mlflow
import wandb
from dataclasses import asdict

from ..models.base import TrajectoryPredictor
from ..models.linear import LinearExtrapolationPredictor
from ..models.kalman import KalmanFilterPredictor
from ..models.lstm import LSTMPredictor
from ..models.transformer import TransformerPredictor
from ..data.loader import SCATDataLoader
from .metrics import calculate_trajectory_metrics, statistical_comparison, bootstrap_confidence_interval
from ..config import settings


class TrajectoryBenchmark:
    """Comprehensive benchmarking framework for trajectory prediction models."""
    
    def __init__(self, data_path: str, results_path: str = None):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path) if results_path else settings.RESULTS_DIR
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = SCATDataLoader(self.data_path)
        
        # Initialize models
        self.models = {
            "linear": LinearExtrapolationPredictor(device=settings.DEVICE),
            "kalman": KalmanFilterPredictor(device=settings.DEVICE),
            "lstm": LSTMPredictor(device=settings.DEVICE),
            "transformer": TransformerPredictor(device=settings.DEVICE)
        }
        
        # Results storage
        self.results = {}
        
    def prepare_data(self, train_ratio: float = 0.8, 
                    sequence_length: int = 20) -> Dict[str, np.ndarray]:
        """
        Prepare training and testing data.
        
        Args:
            train_ratio: Ratio of data for training
            sequence_length: Length of input sequences
        
        Returns:
            Dictionary with train/test splits
        """
        print("Loading and preprocessing SCAT dataset...")
        
        # Load trajectory data
        trajectories = self.data_loader.load_trajectories()
        
        # Create sequences for training
        X, y = self.data_loader.create_sequences(
            trajectories, 
            sequence_length=sequence_length,
            prediction_horizons=settings.PREDICTION_HORIZONS
        )
        
        # Train/test split
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        
        # Temporal split (train on earlier data, test on later)
        indices = np.arange(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        data_splits = {
            "X_train": X[train_indices],
            "y_train": y[train_indices],
            "X_test": X[test_indices],
            "y_test": y[test_indices]
        }
        
        print(f"Data prepared: {len(train_indices)} training, {len(test_indices)} testing samples")
        return data_splits
    
    def train_models(self, data_splits: Dict[str, np.ndarray]) -> None:
        """Train all models that require training."""
        print("Training models...")
        
        X_train = data_splits["X_train"]
        y_train = data_splits["y_train"]
        
        # Train LSTM
        print("Training LSTM...")
        start_time = time.time()
        self.models["lstm"].fit(X_train, y_train, epochs=50)
        lstm_train_time = time.time() - start_time
        print(f"LSTM training completed in {lstm_train_time:.2f} seconds")
        
        # Train Transformer
        print("Training Transformer...")
        start_time = time.time()
        self.models["transformer"].fit(X_train, y_train, epochs=50)
        transformer_train_time = time.time() - start_time
        print(f"Transformer training completed in {transformer_train_time:.2f} seconds")
        
        # Save training times
        self.results["training_times"] = {
            "lstm": lstm_train_time,
            "transformer": transformer_train_time
        }
    
    def evaluate_models(self, data_splits: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate all models on test data."""
        print("Evaluating models...")
        
        X_test = data_splits["X_test"]
        y_test = data_splits["y_test"]
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            model_results = {}
            
            # Evaluate at different horizons
            for horizon in settings.PREDICTION_HORIZONS:
                horizon_results = []
                
                # Batch evaluation for efficiency
                batch_size = 32
                n_batches = len(X_test) // batch_size
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(X_test))
                    
                    X_batch = X_test[start_idx:end_idx]
                    y_batch = y_test[start_idx:end_idx, :horizon, :]  # Select horizon
                    
                    # Predict
                    prediction_result = model.predict(X_batch, horizon)
                    
                    # Calculate metrics
                    metrics = calculate_trajectory_metrics(
                        prediction_result.predictions,
                        y_batch,
                        prediction_result.computation_time,
                        prediction_result.confidence
                    )
                    
                    horizon_results.append(metrics)
                
                model_results[f"{horizon}min"] = horizon_results
            
            results[model_name] = model_results
        
        self.results["evaluation"] = results
        return results
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report."""
        print("Generating comparison report...")
        
        comparison_data = []
        
        for model_name, model_results in self.results["evaluation"].items():
            model_info = self.models[model_name].get_model_info()
            
            for horizon, metrics_list in model_results.items():
                # Calculate aggregate statistics
                horizontal_rmse_values = [m.horizontal_rmse for m in metrics_list]
                vertical_rmse_values = [m.vertical_rmse for m in metrics_list]
                computation_times = [m.computation_time for m in metrics_list]
                
                # Bootstrap confidence intervals
                h_rmse_ci = bootstrap_confidence_interval(horizontal_rmse_values)
                v_rmse_ci = bootstrap_confidence_interval(vertical_rmse_values)
                
                comparison_data.append({
                    "Model": model_name.title(),
                    "Horizon": horizon,
                    "Horizontal RMSE (m)": f"{np.mean(horizontal_rmse_values):.1f}",
                    "Horizontal RMSE CI": f"[{h_rmse_ci[0]:.1f}, {h_rmse_ci[1]:.1f}]",
                    "Vertical RMSE (m)": f"{np.mean(vertical_rmse_values):.1f}",
                    "Vertical RMSE CI": f"[{v_rmse_ci[0]:.1f}, {v_rmse_ci[1]:.1f}]",
                    "Inference Time (ms)": f"{np.mean(computation_times)*1000:.1f}",
                    "Parameters": model_info["parameters"],
                    "Training Required": model_info["training_required"]
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Save report
        report_path = self.results_path / "comparison_report.csv"
        df.to_csv(report_path, index=False)
        print(f"Comparison report saved to {report_path}")
        
        return df
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis between models."""
        print("Performing statistical analysis...")
        
        statistical_results = {}
        
        # Compare each model pair
        model_names = list(self.models.keys())
        
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                comparison_key = f"{model_a}_vs_{model_b}"
                statistical_results[comparison_key] = {}
                
                for horizon in settings.PREDICTION_HORIZONS:
                    horizon_key = f"{horizon}min"
                    
                    metrics_a = self.results["evaluation"][model_a][horizon_key]
                    metrics_b = self.results["evaluation"][model_b][horizon_key]
                    
                    # Statistical comparison
                    stats_result = statistical_comparison(
                        metrics_a, metrics_b, "horizontal_rmse"
                    )
                    
                    statistical_results[comparison_key][horizon_key] = stats_result
        
        self.results["statistical_analysis"] = statistical_results
        return statistical_results
    
    def log_to_mlflow(self) -> None:
        """Log results to MLflow."""
        print("Logging results to MLflow...")
        
        for model_name, model_results in self.results["evaluation"].items():
            with mlflow.start_run(run_name=f"atlas_{model_name}"):
                # Log model info
                model_info = self.models[model_name].get_model_info()
                mlflow.log_params(model_info)
                
                # Log metrics for each horizon
                for horizon, metrics_list in model_results.items():
                    horizontal_rmse_values = [m.horizontal_rmse for m in metrics_list]
                    vertical_rmse_values = [m.vertical_rmse for m in metrics_list]
                    computation_times = [m.computation_time for m in metrics_list]
                    
                    mlflow.log_metric(f"horizontal_rmse_{horizon}", np.mean(horizontal_rmse_values))
                    mlflow.log_metric(f"vertical_rmse_{horizon}", np.mean(vertical_rmse_values))
                    mlflow.log_metric(f"inference_time_{horizon}", np.mean(computation_times))
                
                # Log training time if available
                if model_name in self.results.get("training_times", {}):
                    mlflow.log_metric("training_time", self.results["training_times"][model_name])
    
    def log_to_wandb(self) -> None:
        """Log results to Weights & Biases."""
        print("Logging results to W&B...")
        
        # Initialize W&B project
        wandb.init(project="atlas-trajectory-prediction", name="model_comparison")
        
        # Create comparison table
        comparison_data = []
        
        for model_name, model_results in self.results["evaluation"].items():
            for horizon, metrics_list in model_results.items():
                horizontal_rmse_values = [m.horizontal_rmse for m in metrics_list]
                
                comparison_data.append([
                    model_name,
                    horizon,
                    np.mean(horizontal_rmse_values),
                    np.std(horizontal_rmse_values),
                    len(metrics_list)
                ])
        
        # Log table
        table = wandb.Table(
            data=comparison_data,
            columns=["Model", "Horizon", "Mean RMSE", "Std RMSE", "Samples"]
        )
        wandb.log({"model_comparison": table})
        
        wandb.finish()
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmarking pipeline."""
        print("Starting ATLAS trajectory prediction benchmark...")
        
        # Prepare data
        data_splits = self.prepare_data()
        
        # Train models
        self.train_models(data_splits)
        
        # Evaluate models
        self.evaluate_models(data_splits)
        
        # Generate reports
        comparison_df = self.generate_comparison_report()
        statistical_results = self.statistical_analysis()
        
        # Log to experiment tracking
        self.log_to_mlflow()
        self.log_to_wandb()
        
        # Save complete results
        results_file = self.results_path / "benchmark_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"Benchmark completed! Results saved to {self.results_path}")
        print("\nComparison Summary:")
        print(comparison_df.to_string(index=False))
        
        return self.results
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(asdict(obj))
        else:
            return obj


def main():
    """Main entry point for benchmarking."""
    benchmark = TrajectoryBenchmark(
        data_path=settings.DATA_DIR / "scat",
        results_path=settings.RESULTS_DIR / "experiments"
    )
    
    results = benchmark.run_full_benchmark()
    return results


if __name__ == "__main__":
    main()

