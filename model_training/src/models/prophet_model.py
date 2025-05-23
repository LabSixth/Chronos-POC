from typing import Any, Dict, Optional
import pickle
import pandas as pd
from pathlib import Path
from prophet import Prophet
import logging
import optuna
from src.utils.model_evaluation import compute_metrics

logger = logging.getLogger(__name__)

class ProphetModel:
    """
    Model class for Facebook Prophet time series forecasting with hyperparameter optimization support.
    """
    def __init__(self, config: dict):
        """
        Initialize ProphetModel with config['models']['prophet_model'].
        """
        model_cfg = config['models']['prophet_model']
        self.horizon = model_cfg.get('horizon', 7)
        self.param_grid = model_cfg.get('param_grid', {})
        self.val_size = model_cfg.get('val_size', 0.1)
        self.optimize_objective = [m.upper() for m in model_cfg.get('optimize_objective', ["RMSE", "MAE", "MAPE"])]
        self.n_trials = model_cfg.get('n_trials', 30)
        self.model = None
        self.fitted = False
        self.best_params = None
        self.best_score = None

    def hyperparameter_optimization(self, train_df: pd.DataFrame):
        """
        Perform Bayesian hyperparameter optimization using Optuna and compute_metrics.
        Uses the first metric in optimize_objective as the optimization objective.
        """
        split_idx = int(len(train_df) * (1 - self.val_size))
        train = train_df.iloc[:split_idx]
        val = train_df.iloc[split_idx:]
        def objective(trial):
            params = {}
            for k, v in self.param_grid.items():
                if isinstance(v[0], bool) or isinstance(v[0], str):
                    params[k] = trial.suggest_categorical(k, v)
                else:
                    params[k] = trial.suggest_float(k, min(v), max(v))
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
        logger.info(f"Best params: {self.best_params}, Best score: {self.best_score}")

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the Prophet model to the training data using best_params if available, else prophet_params.
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
        """
        if model is None:
            if not self.fitted:
                raise RuntimeError("Model must be fitted before prediction.")
            model = self.model
        preds = []
        n = len(test_df)
        test_index = test_df.index
        for start in range(0, n, self.horizon):
            end = min(start + self.horizon, n)
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
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving.")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
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
    