"""
Data loading utilities for stock price prediction models.

This module provides functions to load data from local filesystem or S3 storage
using pandas. It handles different data sources based on configuration parameters.
"""

from pathlib import Path
import pandas as pd


def load_parquet_local(local_path: str, column_names=None) -> pd.DataFrame:
    """
    Load a parquet file from the local filesystem.

    Args:
        local_path (str): Relative path to the parquet file from the project root.
        column_names (list, optional): List of columns to read. Reads all columns if None.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    # Assume loader.py is in src/data_processing/, project root is three levels up
    base_dir = Path(__file__).resolve().parent.parent.parent
    file_path = (base_dir / local_path).resolve()
    return pd.read_parquet(file_path, columns=column_names)


def load_parquet_s3(s3_path: str, column_names=None) -> pd.DataFrame:
    """
    Load a parquet file from S3. AWS credentials are read from environment variables
    or ~/.aws/credentials.

    Args:
        s3_path (str): S3 URI to the parquet file.
        column_names (list, optional): List of columns to read. Reads all columns if None.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_parquet(s3_path, columns=column_names, storage_options={"anon": False})


def load_data(config: dict, local=False, use_s3=False) -> pd.DataFrame:
    """
    Load a parquet file according to config, supporting both local and S3 paths.

    Args:
        config (dict): Config dictionary with keys 'local_path', 's3_path', and optionally
            'column_names'.
        local (bool): Whether to load data from local path.
        use_s3 (bool): Whether to load data from S3 path.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        ValueError: If neither local nor use_s3 is specified.
    """
    column_names = config.get("column_names")
    local_path = config.get("local_path")
    s3_path = config.get("s3_path")
    if local:
        return load_parquet_local(local_path, column_names)
    if use_s3:
        return load_parquet_s3(s3_path, column_names)

    raise ValueError("No data source specified. Please specify either local or s3.")
