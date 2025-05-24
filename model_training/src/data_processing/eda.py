"""
Exploratory Data Analysis (EDA) module for time series data.

This module provides automated EDA functionality including:
1. Basic data statistics and information
2. Missing value analysis
3. Time series visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any
import yaml
from .data_loading import load_data
from .data_cleaning import clean_time_series
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

def analyze_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze basic statistics of the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Dict[str, Any]: Dictionary containing basic statistics
    """
    stats = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "basic_stats": df.describe().to_dict(),
        "memory_usage": df.memory_usage(deep=True).to_dict()
    }
    return stats

def analyze_time_series(df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
    """
    Analyze time series specific characteristics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        target_col (str): Name of the target column
        
    Returns:
        Dict[str, Any]: Dictionary containing time series analysis results
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Calculate time series metrics
    ts_stats = {
        "date_range": {
            "start": df[date_col].min().strftime("%Y-%m-%d"),
            "end": df[date_col].max().strftime("%Y-%m-%d")
        },
        "time_gaps": {
            "total_days": (df[date_col].max() - df[date_col].min()).days,
            "missing_days": len(pd.date_range(df[date_col].min(), df[date_col].max())) - len(df)
        },
        "target_stats": {
            "mean": df[target_col].mean(),
            "std": df[target_col].std(),
            "min": df[target_col].min(),
            "max": df[target_col].max()
        }
    }
    return ts_stats

def create_visualizations(df: pd.DataFrame, date_col: str, target_col: str, output_dir: Path) -> None:
    """
    Create and save various visualizations for the time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        target_col (str): Name of the target column
        output_dir (Path): Directory to save the visualizations
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Time series plot
    df = df.sort_values(date_col)
    plt.figure(figsize=(14, 7))
    plt.plot(df[date_col], df[target_col], label=target_col)
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
    sns.histplot(df[target_col], kde=True)
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_plot.png')
    plt.close()
    
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[target_col])
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
    df = load_data(data_cfg, local=not use_s3, s3=use_s3)
    df_clean = clean_time_series(df, config["data_processing"]["data_cleaning"])
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
    
    with open(output_path / "eda_results.yaml", "w") as f:
        yaml.dump(results, f)
    
    logger.info("EDA completed. Results saved to %s", output_path) 