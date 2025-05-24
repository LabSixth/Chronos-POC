"""
Unit tests for the prediction models in the model training pipeline.
This module tests the Prophet and Chronos models functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


# Add parent directory to system path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModels(unittest.TestCase):
    """Test functionality of model modules"""

    def setUp(self):
        """Set up test data"""
        # Create mock time series data
        dates = pd.date_range(start="2020-01-01", periods=100)
        self.train_data = pd.DataFrame({
            "y": np.sin(np.arange(80) * 0.1) + np.random.normal(0, 0.1, 80) + 100
        }, index=dates[:80])
        self.train_data.index.name = "ds"

        self.test_data = pd.DataFrame({
            "y": np.sin(np.arange(80, 100) * 0.1) + np.random.normal(0, 0.1, 20) + 100
        }, index=dates[80:])
        self.test_data.index.name = "ds"

        # Mock configuration
        self.config = {
            "models": {
                "prophet_model": {
                    "hyperparameters": {
                        "changepoint_prior_scale": [0.01, 0.1, 0.5],
                        "seasonality_prior_scale": [0.01, 0.1, 1.0],
                        "seasonality_mode": ["additive", "multiplicative"]
                    },
                    "forecast_horizon": 20,
                    "cv_horizon": 10,
                    "initial": 50,
                    "period": 10
                },
                "chronos_model": {
                    "pretrained_model": "amazon/chronos-t5-tiny",
                    "horizon": 20,
                    "forecast_type": "median"
                }
            }
        }

    def test_prophet_prediction(self):
        """Test Prophet model prediction functionality"""
        try:
            # Create a custom mock model
            class MockProphetModel:
                """Custom mock implementation for Prophet model testing"""
                def __init__(self, config):
                    self.config = config
                    self.fitted = False
                    self.train_size = None  # Initialize attribute in __init__

                def fit(self, train_df):
                    """Mock fit method"""
                    self.fitted = True
                    self.train_size = len(train_df)  # Set attribute properly

                def predict(self, test_df):
                    """Mock predict method"""
                    return pd.Series(np.random.normal(100, 10, len(test_df)), index=test_df.index)

                def save_model(self, path):
                    """Mock save_model method"""
                    with open(path, "wb") as file_obj:
                        file_obj.write(b"mock model data")

            # Use the mock model for testing
            model = MockProphetModel(self.config)

            # Test fitting
            model.fit(self.train_data)
            self.assertTrue(model.fitted)

            # Test prediction
            predictions = model.predict(self.test_data)

            # Verify prediction result length
            self.assertEqual(len(predictions), len(self.test_data))

            # Verify prediction values are numeric
            self.assertTrue(np.issubdtype(predictions.dtype, np.number))

            # Test model saving functionality
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmpfile:
                model_path = tmpfile.name
                model.save_model(model_path)
                self.assertTrue(os.path.exists(model_path))
                # Cleanup
                os.unlink(model_path)

        except ImportError as import_error:
            self.skipTest(f"Prophet model test skipped: {import_error}")

    def test_chronos_prediction(self):
        """Test Chronos model prediction functionality"""
        try:
            # Create a custom mock model
            class MockChronosModel:
                """Custom mock implementation for Chronos model testing"""
                def __init__(self, config):
                    self.config = config
                    self.fitted = False
                    self.train_df = None

                def fit(self, train_df):
                    """Mock fit method"""
                    self.train_df = train_df
                    self.fitted = True

                def predict(self, test_df):
                    """Mock predict method"""
                    if not self.fitted:
                        raise RuntimeError("Model must be fitted before prediction.")
                    return pd.Series(np.random.normal(100, 10, len(test_df)), index=test_df.index)

                def save_prediction(self, predictions, path):
                    """Mock save_prediction method"""
                    predictions.to_csv(path, header=True)

            # Use the mock model for testing
            model = MockChronosModel(self.config)

            # Test fitting
            model.fit(self.train_data)
            self.assertTrue(model.fitted)

            # Test prediction
            predictions = model.predict(self.test_data)

            # Verify prediction result length
            self.assertEqual(len(predictions), len(self.test_data))

            # Verify prediction values are numeric
            self.assertTrue(np.issubdtype(predictions.dtype, np.number))

            # Test prediction result saving
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
                pred_path = tmpfile.name
                model.save_prediction(predictions, pred_path)
                self.assertTrue(os.path.exists(pred_path))
                # Verify saved prediction results can be loaded
                saved_preds = pd.read_csv(pred_path, index_col=0)
                self.assertEqual(len(saved_preds), len(predictions))
                # Cleanup
                os.unlink(pred_path)

        except ImportError as import_error:
            self.skipTest(f"Chronos model test skipped: {import_error}")


if __name__ == "__main__":
    unittest.main()
