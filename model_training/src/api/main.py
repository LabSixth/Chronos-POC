"""
Stock Price Prediction API using FastAPI.

This module provides REST API endpoints for:
1. Making stock price predictions using trained Prophet model
2. Health check endpoint for monitoring
"""

import logging
import yaml

# Third-party imports that may not be available during linting
# pylint: disable=import-error,no-name-in-module
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# pylint: enable=import-error,no-name-in-module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Price Prediction API")


# pylint: disable=too-few-public-methods
class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.

    Attributes:
        date: String date in format YYYY-MM-DD
        symbol: Stock symbol, default is AAPL
    """
    date: str
    symbol: str = "AAPL"


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint.

    Attributes:
        date: Input date
        symbol: Stock symbol
        predicted_price: Model's predicted price
        model: Name of the model used for prediction
    """
    date: str
    symbol: str
    predicted_price: float
    model: str
# pylint: enable=too-few-public-methods


def load_model(model_path: str):
    """
    Load a serialized model from the given path.

    Args:
        model_path: Path to the serialized model file

    Returns:
        The loaded model

    Raises:
        HTTPException: If the model cannot be loaded
    """
    try:
        return joblib.load(model_path)
    except Exception as exception:
        logger.error("Error loading model: %s", exception)
        raise HTTPException(status_code=500, detail="Model loading failed") from exception


@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup.

    Loads configuration and models from disk.
    """
    # Load configuration
    with open("config/default-config.yaml", "r", encoding="utf-8") as file_handle:
        app.config = yaml.safe_load(file_handle)

    # Load models
    app.prophet_model = load_model("artifacts/prophet_model.pkl")
    logger.info("Models loaded successfully")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a price prediction for the given date and symbol.

    Args:
        request: The prediction request containing date and symbol

    Returns:
        PredictionResponse with the predicted price

    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert date string to datetime
        date = pd.to_datetime(request.date)

        # Make prediction using Prophet model
        future = pd.DataFrame({'ds': [date]})
        forecast = app.prophet_model.predict(future)
        predicted_price = float(forecast['yhat'].iloc[0])

        return PredictionResponse(
            date=request.date,
            symbol=request.symbol,
            predicted_price=predicted_price,
            model="prophet"
        )
    except Exception as exception:
        logger.error("Prediction error: %s", exception)
        raise HTTPException(status_code=500, detail=str(exception)) from exception


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.

    Returns:
        Dictionary with status information
    """
    return {"status": "healthy"}
