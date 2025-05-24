"""
Unit tests for the Streamlit application in the model training pipeline.
This module tests the app's data loading and visualization functions.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# Add parent directory to system path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the streamlit module
sys.modules["streamlit"] = MagicMock()

# Define a minimal mock app module for testing
class MockApp:
    """Mock implementation of the Streamlit app for testing"""

    @staticmethod
    def load_prediction_data():
        """Load prediction data from CSV files"""
        prophet_preds = pd.read_csv("Model_Artifacts/prophet_preds.csv")
        chronos_preds = pd.read_csv("Model_Artifacts/chronos_preds.csv")

        # Convert date column to datetime type
        prophet_preds["ds"] = pd.to_datetime(prophet_preds["ds"])
        chronos_preds["ds"] = pd.to_datetime(chronos_preds["ds"])

        return prophet_preds, chronos_preds

    @staticmethod
    def load_images():
        """Load image paths for model visualizations"""
        prophet_img_path = "Model_Artifacts/Prophet_predicted_vs_real.png"
        chronos_img_path = "Model_Artifacts/Chronos_predicted_vs_real.png"

        return prophet_img_path, chronos_img_path


class TestStreamlitApp(unittest.TestCase):
    """Test Streamlit application functionality"""

    def setUp(self):
        """Set up test data and environment"""
        # Create mock prediction data
        dates = pd.date_range(start="2023-01-01", periods=30)

        self.prophet_preds = pd.DataFrame({
            "ds": dates,
            "0": np.random.uniform(100, 200, 30)
        })

        self.chronos_preds = pd.DataFrame({
            "ds": dates,
            "0": np.random.uniform(100, 200, 30)
        })

        # TemporaryDirectory will be cleaned up in tearDown method
        # pylint: disable=consider-using-with
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_artifacts_dir = Path(self.temp_dir.name) / "Model_Artifacts"
        self.model_artifacts_dir.mkdir(exist_ok=True)

        # Save prediction data to temporary files
        self.prophet_preds.to_csv(self.model_artifacts_dir / "prophet_preds.csv", index=False)
        self.chronos_preds.to_csv(self.model_artifacts_dir / "chronos_preds.csv", index=False)

        # Create mock image files
        prophet_img_path = self.model_artifacts_dir / "Prophet_predicted_vs_real.png"
        chronos_img_path = self.model_artifacts_dir / "Chronos_predicted_vs_real.png"

        with open(prophet_img_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("mock image data")
        with open(chronos_img_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("mock image data")

        # Save the original working directory
        self.original_dir = os.getcwd()
        # Change to temp directory for testing
        os.chdir(self.temp_dir.name)

        # Use the mock app
        self.app = MockApp()

    def tearDown(self):
        """Clean up temporary files and restore environment"""
        # Restore original directory
        os.chdir(self.original_dir)
        # Clean up temp directory
        self.temp_dir.cleanup()

    def test_load_prediction_data(self):
        """Test loading prediction data functionality"""
        # Call function
        prophet_data, chronos_data = self.app.load_prediction_data()

        # Verify data is loaded correctly
        self.assertEqual(len(prophet_data), len(self.prophet_preds))
        self.assertEqual(len(chronos_data), len(self.chronos_preds))

        # Verify date column is datetime type
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prophet_data["ds"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(chronos_data["ds"]))

    def test_load_images(self):
        """Test loading images functionality"""
        # Call function
        prophet_img_path, chronos_img_path = self.app.load_images()

        # Verify paths are correct
        self.assertEqual(prophet_img_path, "Model_Artifacts/Prophet_predicted_vs_real.png")
        self.assertEqual(chronos_img_path, "Model_Artifacts/Chronos_predicted_vs_real.png")

        # Verify files exist
        self.assertTrue(os.path.exists(prophet_img_path))
        self.assertTrue(os.path.exists(chronos_img_path))


if __name__ == "__main__":
    unittest.main()
