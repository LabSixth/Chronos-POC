"""
Chronos model implementation for time series forecasting.

This module provides a wrapper around Amazon's Chronos, a zero-shot time series
forecasting model built on T5. It handles model loading, prediction generation,
and result saving.
"""

from pathlib import Path

import pandas as pd
import numpy as np

# Suppress import errors for optional dependencies with pylint disable
# pylint: disable=import-error,no-member
try:
    import torch
    from chronos import ChronosPipeline
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    # Create dummy torch module for type checking
    class DummyTorch:
        """Dummy torch module to avoid import errors"""
        # pylint: disable=unused-argument
        def tensor(self, *args, **kwargs):
            """Dummy tensor method"""
            return np.array([])
        def from_numpy(self, *args, **kwargs):
            """Dummy from_numpy method"""
            return np.array([])
        # pylint: enable=unused-argument
    torch = DummyTorch()
# pylint: enable=import-error,no-member


class ChronosModel:
    """
    Model class for Amazon Chronos zero-shot time series forecasting.
    This implementation uses the ChronosPipeline for probabilistic forecasting.
    All parameters are read from config['models']['chronos_model'].
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize the Chronos pipeline using config['models']['chronos_model'].

        Args:
            config (dict): Full config dictionary loaded from YAML.
        """
        if not HAS_DEPENDENCIES:
            raise ImportError(
                "Chronos dependencies not installed. "
                "Please install with 'pip install chronos-forecast'"
            )

        model_cfg = config['models']['chronos_model']
        self.horizon = model_cfg.get('horizon', 7)
        model_name = model_cfg.get('pretrained_model', 'amazon/chronos-t5-tiny')
        device_map = model_cfg.get('device_map', 'auto')
        torch_dtype = model_cfg.get('torch_dtype', 'bfloat16')
        self.forecast_type = model_cfg.get('forecast_type', 'median')
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        self.fitted = False
        self.train_df = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Prepare for prediction (no-op for Chronos as it's zero-shot).

        Args:
            train_df (pd.DataFrame): Training data with 'ds' index and 'y' column
        """
        self.train_df = train_df
        self.fitted = True

    # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks
    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Perform rolling multi-step forecasting on the test set using a fixed fitted model.

        For each window, predict 'horizon' steps ahead, sliding the window until
        the test set is covered.

        Args:
            test_df (pd.DataFrame): Test set with 'ds' index and 'y' column.

        Returns:
            pd.Series: Predicted values aligned with the test set index.

        Raises:
            RuntimeError: If model is not fitted before prediction.
            ValueError: If forecast_type is not supported.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        preds = []
        num_samples = len(test_df)
        test_index = test_df.index
        for start in range(0, num_samples, self.horizon):
            end = min(start + self.horizon, num_samples)
            # Prepare the input series
            # Use all available data up to this window
            series_data = self.train_df['y'].tolist() + test_df['y'].iloc[:start].tolist()

            # Convert to numpy array first (works in all scenarios)
            np_series = np.array(series_data, dtype=np.float32)

            # Try to convert to torch tensor if dependencies available
            if HAS_DEPENDENCIES:
                # pylint: disable=no-member
                try:
                    if hasattr(torch, 'tensor'):
                        series = torch.tensor(np_series)
                    elif hasattr(torch, 'FloatTensor'):
                        series = torch.FloatTensor(np_series)
                    else:
                        series = torch.from_numpy(np_series)
                except (AttributeError, TypeError):
                    series = np_series
                # pylint: enable=no-member
            else:
                series = np_series

            forecast = self.pipeline.predict(series, end - start)

            # Extract prediction values based on forecast type
            if HAS_DEPENDENCIES:
                # pylint: disable=no-member
                try:
                    if self.forecast_type == 'mean':
                        values = forecast[0].mean(dim=0).numpy()
                    elif self.forecast_type in ('quantile', 'median'):
                        # Try different methods to get median value depending on torch version
                        try:
                            if hasattr(torch, 'quantile'):
                                values = torch.quantile(forecast[0], 0.5, dim=0).numpy()
                            elif hasattr(torch, 'median'):
                                values = torch.median(forecast[0], dim=0)[0].numpy()
                            else:
                                # Manual median calculation
                                sorted_forecast = forecast[0].sort(dim=0)[0]
                                mid_idx = sorted_forecast.shape[0] // 2
                                values = sorted_forecast[mid_idx].numpy()
                        except (AttributeError, TypeError):
                            # Convert to numpy and calculate median
                            values = np.median(forecast[0].numpy(), axis=0)
                    else:
                        raise ValueError(f"Unsupported forecast_type: {self.forecast_type}")
                except (AttributeError, TypeError):
                    # If torch methods fail, use numpy directly
                    forecast_np = forecast[0].numpy()
                    if self.forecast_type == 'mean':
                        values = np.mean(forecast_np, axis=0)
                    else:  # default to median
                        values = np.median(forecast_np, axis=0)
                # pylint: enable=no-member
            else:
                # Fallback to numpy when torch is not available
                forecast_np = np.array(forecast[0])
                if self.forecast_type == 'mean':
                    values = np.mean(forecast_np, axis=0)
                else:  # default to median
                    values = np.median(forecast_np, axis=0)

            preds.extend(values[:end - start])
        preds = pd.Series(preds, index=test_index)
        return preds
    # pylint: enable=too-many-branches,too-many-statements,too-many-nested-blocks

    def save_prediction(self, predictions: pd.Series, path: str) -> None:
        """
        Save the predicted values to a CSV file at the given path.

        Args:
            predictions (pd.Series): Predicted values to save.
            path (str): File path to save the predictions.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(path, header=True)
