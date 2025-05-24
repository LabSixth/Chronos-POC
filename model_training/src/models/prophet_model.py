"""
Prophet model implementation for time series forecasting.

This module provides a wrapper around Facebook's Prophet for time series forecasting.
It includes hyperparameter optimization via Optuna, model fitting, prediction generation,
and model/prediction saving functionality.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
# Third-party imports with error handling
try:
    from prophet import Prophet
    import optuna
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

# pylint: disable=import-error
from src.utils.model_evaluation import compute_metrics
# pylint: enable=import-error

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class ProphetModel:
    """
    Model class for Facebook Prophet time series forecasting with
    hyperparameter optimization support.
    """
    def __init__(self, config: dict):
        """
        Initialize ProphetModel with config['models']['prophet_model'].

        Args:
            config (dict): Full configuration dictionary loaded from YAML.
        """
        if not HAS_DEPENDENCIES:
            raise ImportError(
                "Prophet dependencies not installed. "
                "Please install with 'pip install prophet optuna'"
            )

        model_cfg = config['models']['prophet_model']
        self.horizon = model_cfg.get('horizon', 7)
        self.param_grid = model_cfg.get('param_grid', {})
        self.val_size = model_cfg.get('val_size', 0.1)
        self.optimize_objective = [
            m.upper() for m in model_cfg.get('optimize_objective', ["RMSE", "MAE", "MAPE"])
        ]
        self.n_trials = model_cfg.get('n_trials', 30)
        self.model = None
        self.fitted = False
        self.best_params = None
        self.best_score = None
        # Default prophet parameters if no optimization is performed
        self.prophet_params = model_cfg.get('prophet_params', {})

    def hyperparameter_optimization(self, train_df: pd.DataFrame):
        """
        Perform Bayesian hyperparameter optimization using Optuna and compute_metrics.
        Uses the first metric in optimize_objective as the optimization objective.

        Args:
            train_df (pd.DataFrame): Training data with 'ds' index and 'y' column
        """
        split_idx = int(len(train_df) * (1 - self.val_size))
        train = train_df.iloc[:split_idx]
        val = train_df.iloc[split_idx:]

        def objective(trial):
            params = {}
            for param_name, param_values in self.param_grid.items():
                if isinstance(param_values[0], (bool, str)):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
            model = Prophet(**params)
            model.fit(train.reset_index())
            preds = self.predict(val, model=model)
            score_dict = compute_metrics(val['y'], preds, self.optimize_objective)
            score = score_dict[self.optimize_objective[0]]
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_score = study.best_value
        logger.info("Best params: %s, Best score: %s", self.best_params, self.best_score)

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the Prophet model to the training data using best_params if available,
        else prophet_params.

        Args:
            train_df (pd.DataFrame): Training data with 'ds' index and 'y' column
        """
        params = self.best_params or self.prophet_params
        self.model = Prophet(**params)
        self.model.fit(train_df.reset_index())
        self.fitted = True
        logger.info("Prophet model fitted successfully.")

    def predict(self, test_df: pd.DataFrame, model: Prophet = None) -> pd.Series:
        """
        Rolling multi-step forecast for the test set using the fitted model or a provided model.

        Args:
            test_df (pd.DataFrame): Test set with 'ds' index and 'y' column.
            model (Prophet, optional): Prophet model to use for prediction. Defaults to self.model.

        Returns:
            pd.Series: Predicted values aligned with the test set index.

        Raises:
            RuntimeError: If model is not fitted and no model is provided.
        """
        if model is None:
            if not self.fitted:
                raise RuntimeError("Model must be fitted before prediction.")
            model = self.model

        preds = []
        num_samples = len(test_df)
        test_index = test_df.index

        for start in range(0, num_samples, self.horizon):
            end = min(start + self.horizon, num_samples)
            # Create future dataframe for prediction
            future = model.make_future_dataframe(
                periods=end - start,
                freq='D',
                include_history=False
            )
            forecast = model.predict(future)
            preds.extend(forecast['yhat'].values[:end - start])

        return pd.Series(preds, index=test_index)

    def save_model(self, path: str) -> None:
        """
        Save the Prophet model to a pickle file at the given path.

        Args:
            path (str): File path to save the model.

        Raises:
            RuntimeError: If model is not fitted before saving.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving.")

        with open(path, 'wb') as file_handle:
            pickle.dump(self.model, file_handle)

        logger.info("Prophet model saved to %s", path)

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
        logger.info("Predictions saved to %s", path)
