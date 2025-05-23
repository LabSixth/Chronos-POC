from typing import Any, Dict, Optional
import pandas as pd
import torch
from chronos import ChronosPipeline
from pathlib import Path

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

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Perform rolling multi-step forecasting on the test set using a fixed fitted model.
        For each window, predict 'horizon' steps ahead, sliding the window until the test set is covered.
        Args:
            test_df (pd.DataFrame): Test set with 'ds' index and 'y' column.
        Returns:
            pd.Series: Predicted values aligned with the test set index.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        preds = []
        n = len(test_df)
        test_index = test_df.index
        for start in range(0, n, self.horizon):
            end = min(start + self.horizon, n)
            # Prepare the input series as a torch tensor (use all available data up to this window)
            series = torch.tensor(self.train_df['y'].tolist() + test_df['y'].iloc[:start].tolist(), dtype=torch.float32)
            forecast = self.pipeline.predict(series, end - start)
            if self.forecast_type == 'mean':
                values = forecast[0].mean(dim=0).numpy()
            elif self.forecast_type == 'quantile' or self.forecast_type == 'median':
                values = torch.quantile(forecast[0], 0.5, dim=0).numpy()
            else:
                raise ValueError(f"Unsupported forecast_type: {self.forecast_type}")
            preds.extend(values[:end - start])
        preds = pd.Series(preds, index=test_index)
        return preds

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
