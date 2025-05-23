from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import logging
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Price Prediction API")

class PredictionRequest(BaseModel):
    date: str
    symbol: str = "AAPL"

class PredictionResponse(BaseModel):
    date: str
    symbol: str
    predicted_price: float
    model: str

def load_model(model_path: str):
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.on_event("startup")
async def startup_event():
    # Load configuration
    with open("config/default-config.yaml", "r") as f:
        app.config = yaml.safe_load(f)
    
    # Load models
    app.prophet_model = load_model("artifacts/prophet_model.pkl")
    logger.info("Models loaded successfully")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
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
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 