"""
Model evaluation utilities for time series forecasting.

This module provides functions for:
1. Computing performance metrics for model evaluation
2. Saving metrics to files
3. Visualizing predicted vs. actual values
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, metrics: List[str]) -> Dict[str, float]:
    """
    Compute selected metrics between true and predicted values.

    Args:
        y_true (pd.Series): Ground truth values.
        y_pred (pd.Series): Predicted values.
        metrics (List[str]): List of metric names to compute based on the config.

    Returns:
        Dict[str, float]: Dictionary with selected metrics.
    """
    results = {}
    for metric in metrics:
        if metric.upper() == "RMSE":
            results["RMSE"] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric.upper() == "MAE":
            results["MAE"] = np.mean(np.abs(y_true - y_pred))
        elif metric.upper() == "MAPE":
            # Replace zeros with NaN to avoid division by zero
            denominator = y_true.replace(0, np.nan)
            results["MAPE"] = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return results


def evaluate_from_config(y_true: pd.Series, y_pred: pd.Series, config: dict) -> Dict[str, float]:
    """
    Evaluate metrics as specified in the config.

    Args:
        y_true (pd.Series): Ground truth values.
        y_pred (pd.Series): Predicted values.
        config (dict): Config dictionary containing 'metrics' key.

    Returns:
        Dict[str, float]: Dictionary with selected metrics.
    """
    metrics = config.get('metrics', ['RMSE', 'MAE'])
    return compute_metrics(y_true, y_pred, metrics)


def save_metrics(metrics: dict, path: str) -> None:
    """
    Save the metrics dictionary as a YAML file to the given path.

    Args:
        metrics (dict): Metrics to save.
        path (str): File path to save the YAML.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        yaml.dump(metrics, file_handle)


def prediction_visualization(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    artifacts_dir: str
) -> None:
    """
    Plot predicted vs real close price and save the figure in the artifacts directory.

    Args:
        y_true (pd.Series): Ground truth values.
        y_pred (pd.Series): Predicted values.
        model_name (str): Name of the model.
        artifacts_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label="Real Close Price", color="blue")
    plt.plot(y_true.index, y_pred.values, label="Predicted Close Price", color="orange")
    plt.title(f"{model_name}: Predicted vs Real Close Price")
    plt.xlabel("Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    # Save figure to specified path
    plot_filename = f"{model_name}_predicted_vs_real.png"
    plot_path = Path(artifacts_dir) / plot_filename
    plt.savefig(plot_path)
    plt.close()
