"""
Unit tests for the data processing module in the model training pipeline.
This module tests the data cleaning and train-test split functionality.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.data_processing.data_cleaning import clean_time_series, train_test_split

# Add parent directory to system path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataProcessing(unittest.TestCase):
    """Test functionality of data processing module"""

    def setUp(self):
        """Set up test data"""
        # Create mock stock price data
        dates = pd.date_range(start="2020-01-01", periods=100)
        self.mock_data = pd.DataFrame({
            "Date": dates,
            "Open": np.random.uniform(100, 200, 100),
            "High": np.random.uniform(110, 210, 100),
            "Low": np.random.uniform(90, 190, 100),
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.randint(1000, 10000, 100)
        })

        # Mock configuration
        self.data_config = {
            "filepath": "dummy_path.csv",
            "date_column": "Date",
            "target_column": "Close"
        }

        self.clean_config = {
            "date_column": "Date",
            "target_column": "Close",
            "rename_cols": {"Date": "ds", "Close": "y"},
            "test_size": 0.2,
            "random_state": 42
        }

    def test_clean_time_series(self):
        """Test time series data cleaning"""
        df_clean = clean_time_series(self.mock_data, self.clean_config)

        # Check if index name is 'ds'
        self.assertEqual(df_clean.index.name, "ds")

        # Check if target column is renamed to 'y'
        self.assertIn("y", df_clean.columns)

        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_clean.index))

        # Check data volume remains unchanged
        self.assertEqual(len(df_clean), len(self.mock_data))

    def test_train_test_split(self):
        """Test train and test set split"""
        # Clean data first
        df_clean = clean_time_series(self.mock_data, self.clean_config)

        # Split into train and test sets
        train_df, test_df = train_test_split(df_clean, self.clean_config)

        # Verify split size
        expected_test_size = int(len(df_clean) * self.clean_config["test_size"])
        self.assertEqual(len(test_df), expected_test_size)
        self.assertEqual(len(train_df), len(df_clean) - expected_test_size)

        # Verify date chronological order (train set should be before test set)
        self.assertTrue(train_df.index.max() < test_df.index.min())


if __name__ == "__main__":
    unittest.main()
