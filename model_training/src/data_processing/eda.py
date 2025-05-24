"""
Exploratory Data Analysis (EDA) module for time series data.

This module provides automated EDA functionality including:
1. Basic data statistics and information
2. Missing value analysis
3. Time series visualization
"""

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# use relative import to avoid issues with Docker environment
from .data_cleaning import clean_time_series
from .data_loading import load_data

logger = logging.getLogger(__name__)


def analyze_basic_stats(data_frame: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze basic statistics of the dataset.

    Args:
        data_frame (pd.DataFrame): Input DataFrame

    Returns:
        Dict[str, Any]: Dictionary containing basic statistics
    """
    stats = {
        "shape": data_frame.shape,
        "dtypes": data_frame.dtypes.to_dict(),
        "missing_values": data_frame.isnull().sum().to_dict(),
        "basic_stats": data_frame.describe().to_dict(),
        "memory_usage": data_frame.memory_usage(deep=True).to_dict()
    }
    return stats


def analyze_time_series(data_frame: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
    """
    Analyze time series specific characteristics.

    Args:
        data_frame (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        target_col (str): Name of the target column

    Returns:
        Dict[str, Any]: Dictionary containing time series analysis results
    """
    data_frame = data_frame.copy()
    data_frame[date_col] = pd.to_datetime(data_frame[date_col])
    data_frame = data_frame.sort_values(date_col)

    # Calculate time series metrics
    ts_stats = {
        "date_range": {
            "start": data_frame[date_col].min().strftime("%Y-%m-%d"),
            "end": data_frame[date_col].max().strftime("%Y-%m-%d")
        },
        "time_gaps": {
            "total_days": (data_frame[date_col].max() - data_frame[date_col].min()).days,
            "missing_days": (
                len(pd.date_range(data_frame[date_col].min(), data_frame[date_col].max())) -
                len(data_frame)
            )
        },
        "target_stats": {
            "mean": data_frame[target_col].mean(),
            "std": data_frame[target_col].std(),
            "min": data_frame[target_col].min(),
            "max": data_frame[target_col].max()
        }
    }
    return ts_stats


def create_visualizations(
    data_frame: pd.DataFrame,
    date_col: str,
    target_col: str,
    output_dir: Path
) -> None:
    """
    Create and save various visualizations for the time series data.

    Args:
        data_frame (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        target_col (str): Name of the target column
        output_dir (Path): Directory to save the visualizations
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Time series plot
    data_frame = data_frame.sort_values(date_col)
    plt.figure(figsize=(14, 7))
    plt.plot(data_frame[date_col], data_frame[target_col], label=target_col)
    plt.title(f'Time Series Plot of {target_col}')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'time_series_plot.png')
    plt.close()

    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data_frame[target_col], kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_plot.png')
    plt.close()

    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data_frame[target_col])
    plt.title(f'Box Plot of {target_col}')
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig(output_dir / 'box_plot.png')
    plt.close()


def run_eda(config: dict, output_dir: str = "artifacts/eda") -> None:
    """
    Run complete EDA pipeline.

    Args:
        config (dict): Configuration dictionary
        output_dir (str): Directory to save EDA results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and clean data
    data_cfg = config["data_processing"]["data_loading"]
    use_s3 = data_cfg.get("use_s3", False)
    data_frame = load_data(data_cfg, local=not use_s3, use_s3=use_s3)
    df_clean = clean_time_series(data_frame, config["data_processing"]["data_cleaning"])
    df_for_eda = df_clean.reset_index()  # columns: ds, y

    # Run analyses
    basic_stats = analyze_basic_stats(df_for_eda)
    ts_stats = analyze_time_series(df_for_eda, "ds", "y")

    # Create visualizations
    create_visualizations(df_for_eda, "ds", "y", output_path)

    # Save analysis results
    results = {
        "basic_statistics": basic_stats,
        "time_series_analysis": ts_stats
    }

    with open(output_path / "eda_results.yaml", "w", encoding="utf-8") as file_handle:
        yaml.dump(results, file_handle)

    logger.info("EDA completed. Results saved to %s", output_path)
