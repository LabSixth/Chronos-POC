"""
Time series data cleaning and preparation module.

This module provides functionality for:
1. Basic time series data cleaning and preparation
2. Train-test splitting for time series data
3. Saving processed data to artifacts
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

def clean_time_series(data_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean and prepare time series data for single-series models.
    Renames columns to match model requirements (date -> ds, target -> y).
    
    Args:
        data_frame (pd.DataFrame): Input DataFrame with raw data
        config (dict): Config dictionary with keys 'date_column' and 'target_column'
        
    Returns:
        pd.DataFrame: Processed DataFrame with columns 'ds' and 'y'
    """
    date_column = config.get("date_column", "from")
    target_column = config.get("target_column", "close")
    data_frame = data_frame.copy()
    # convert date column to datetime
    data_frame[date_column] = pd.to_datetime(data_frame[date_column])
    # set date column as index
    data_frame = data_frame.set_index(date_column)[[target_column]]
    data_frame = data_frame.sort_index()  # ensure that the data is sorted by date
    data_frame = data_frame.rename(columns={target_column: "y"})  # rename target column to y
    data_frame.index.name = "ds" # rename index to ds
    return data_frame

def train_test_split(data_frame: pd.DataFrame, config: dict):
    """
    Split time series data into train and test sets using config.
    
    Args:
        data_frame (pd.DataFrame): Input DataFrame with processed data
        config (dict): Config dictionary with 'test_size' parameter
        
    Returns:
        tuple: (train_df, test_df) DataFrames containing train and test data
    """
    # Get split parameters from config
    test_size = config.get('test_size', 0.2)
    # Calculate split index
    split_idx = int(len(data_frame) * (1 - test_size))
    # Split the data
    train_df = data_frame.iloc[:split_idx]
    test_df = data_frame.iloc[split_idx:]
    logger.info(
        "Split data into train (%d samples) and test (%d samples) sets",
        len(train_df),
        len(test_df)
    )
    return train_df, test_df

def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path: str,
    prefix: str = "processed"
) -> None:
    """
    Save processed training and testing data to the artifacts directory.
    
    Args:
        train_df (pd.DataFrame): Processed training data
        test_df (pd.DataFrame): Processed testing data
        path (str): Directory to save the processed data
        prefix (str): Prefix for the saved files
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)

    # Save training data
    train_path = path / f"{prefix}_train.csv"
    train_df.to_csv(train_path)
    logger.info("Saved processed training data to %s", train_path)
    # Save testing data
    test_path = path / f"{prefix}_test.csv"
    test_df.to_csv(test_path)
    logger.info("Saved processed testing data to %s", test_path)

    # Save data summary
    summary = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_date_range": {
            "start": train_df.index.min().strftime("%Y-%m-%d"),
            "end": train_df.index.max().strftime("%Y-%m-%d")
        },
        "test_date_range": {
            "start": test_df.index.min().strftime("%Y-%m-%d"),
            "end": test_df.index.max().strftime("%Y-%m-%d")
        }
    }
    summary_path = path / f"{prefix}_summary.yaml"
    with open(summary_path, 'w', encoding='utf-8') as file_handle:
        yaml.dump(summary, file_handle)
    logger.info("Saved data summary to %s", summary_path)
