from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from pathlib import Path

# Get the absolute path to the data directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

# Load model and feature columns
try:
    model = joblib.load(DATA_DIR / 'rf_model.pkl')
    feature_columns = joblib.load(DATA_DIR / 'feature_columns.pkl')
except Exception as e:
    print(f"Error loading model or feature columns: {e}")
    model = None
    feature_columns = None

class AQIInput(BaseModel):
    # Define all input fields (example)
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    temp: float
    rhum: float
    pres: float
    wspd: float
    hour: int
    day: int
    weekday: int
    month: int
    AQI_change_1h: float
    AQI_rolling_mean_6h: float
    # Add other fields as needed

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ShineAQI API is running"}

@app.get("/health")
def health_check():
    if model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model or feature columns not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_aqi(data: AQIInput):
    if model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model or feature columns not loaded")
    
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the features used by the model
        df = df[feature_columns]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return {
            "predicted_aqi": float(prediction),
            "input_features": data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 