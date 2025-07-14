"""Evaluation metrics for trajectory prediction."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import scipy.stats as stats


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    horizontal_rmse: float
    vertical_rmse: float
    horizontal_mae: float
    vertical_mae: float
    max_horizontal_error: float
    max_vertical_error: float
    computation_time: float
    confidence_score: float


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, 
                      lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Calculate haversine distance between two points.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_trajectory_metrics(predictions: np.ndarray, 
                               ground_truth: np.ndarray,
                               computation_time: float,
                               confidence: float) -> EvaluationMetrics:
    """
    Calculate comprehensive trajectory prediction metrics.
    
    Args:
        predictions: Predicted positions [batch_size, horizon, 3] (lat, lon, alt)
        ground_truth: True positions [batch_size, horizon, 3]
        computation_time: Time taken for prediction
        confidence: Model confidence score
    
    Returns:
        EvaluationMetrics object
    """
    # Extract coordinates
    pred_lat, pred_lon, pred_alt = predictions[:, :, 0], predictions[:, :, 1], predictions[:, :, 2]
    true_lat, true_lon, true_alt = ground_truth[:, :, 0], ground_truth[:, :, 1], ground_truth[:, :, 2]
    
    # Calculate horizontal errors (haversine distance)
    horizontal_errors = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
    
    # Calculate vertical errors (altitude difference in meters)
    vertical_errors = np.abs(pred_alt - true_alt)
    
    # Calculate metrics
    horizontal_rmse = np.sqrt(np.mean(horizontal_errors**2))
    vertical_rmse = np.sqrt(np.mean(vertical_errors**2))
    horizontal_mae = np.mean(horizontal_errors)
    vertical_mae = np.mean(vertical_errors)
    max_horizontal_error = np.max(horizontal_errors)
    max_vertical_error = np.max(vertical_errors)
    
    return EvaluationMetrics(
        horizontal_rmse=horizontal_rmse,
        vertical_rmse=vertical_rmse,
        horizontal_mae=horizontal_mae,
        vertical_mae=vertical_mae,
        max_horizontal_error=max_horizontal_error,
        max_vertical_error=max_vertical_error,
        computation_time=computation_time,
        confidence_score=confidence
    )


def statistical_comparison(metrics_a: List[EvaluationMetrics], 
                         metrics_b: List[EvaluationMetrics],
                         metric_name: str = "horizontal_rmse") -> Dict[str, float]:
    """
    Perform statistical comparison between two sets of metrics.
    
    Args:
        metrics_a: First set of metrics
        metrics_b: Second set of metrics
        metric_name: Name of metric to compare
    
    Returns:
        Dictionary with statistical test results
    """
    values_a = [getattr(m, metric_name) for m in metrics_a]
    values_b = [getattr(m, metric_name) for m in metrics_b]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((np.std(values_a)**2 + np.std(values_b)**2) / 2))
    cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std
    
    # Wilcoxon signed-rank test (non-parametric)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(values_a, values_b)
    
    return {
        "mean_a": np.mean(values_a),
        "mean_b": np.mean(values_b),
        "std_a": np.std(values_a),
        "std_b": np.std(values_b),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_value": wilcoxon_p,
        "significant": p_value < 0.05
    }


def bootstrap_confidence_interval(values: List[float], 
                                confidence_level: float = 0.95,
                                n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Args:
        values: List of values
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_means = []
    n = len(values)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_bound, upper_bound

