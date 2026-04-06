from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import pandas as pd
import numpy as np
import logging
import uvicorn
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
try:
    model = joblib.load("diamonds_model.joblib")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Загрузка PowerTransformer
try:
    power_trans = joblib.load("power_transformer.joblib")
    logger.info("PowerTransformer loaded successfully")
except Exception as e:
    logger.error(f"Error loading PowerTransformer: {e}")
    power_trans = None

# Загрузка препроцессора
try:
    preprocessor = joblib.load("preprocessor.joblib")
    logger.info("Preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Error loading preprocessor: {e}")
    preprocessor = None

app = FastAPI(title="Diamonds Price Prediction API")


class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float


def create_features(df):
    """Создание новых признаков"""
    df = df.copy()
    df['aspect_ratio'] = df['x'] / (df['y'] + 0.001)
    df['volume'] = df['x'] * df['y'] * df['z']
    df['table_area'] = (df['table'] / 100) * (df['x'] * df['y'])
    df['depth_ratio'] = df['depth'] / (df['z'] + 0.001)
    return df


@app.post("/predict", summary="Predict diamond price")
async def predict(diamond: DiamondFeatures):
    """Предсказывает стоимость алмаза"""
    try:
        input_data = pd.DataFrame([diamond.dict()])
        
        input_data = create_features(input_data)

        if preprocessor is not None:
            input_processed = preprocessor.transform(input_data)
        else:
            return {"error": "Preprocessor not loaded"}
        
        pred_transformed = model.predict(input_processed)[0]

        if power_trans is not None:
            price = power_trans.inverse_transform([[pred_transformed]])[0][0]
        else:
            price = pred_transformed
        
        price = round(float(price), 2)
        logger.info(f"Predicted price: ${price}")
        
        return {"predicted_price": price, "currency": "USD"}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)