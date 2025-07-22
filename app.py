# app.py

from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from src.preprocess import preprocess_data_for_prediction

app = FastAPI()
model = joblib.load("artifacts/model.pkl")


@app.get("/")
async def health_check():
    return {"status": "running", "message": "API is live and serving predictions"}



@app.post("/predict")
async def predict(file_telemetry: UploadFile = File(...), file_machines: UploadFile = File(...)):
    telemetry = pd.read_csv(file_telemetry.file, parse_dates=["datetime"])
    machines = pd.read_csv(file_machines.file)

    df = preprocess_data_for_prediction(telemetry, machines)
    input_data = df.drop(columns=["datetime"], errors="ignore")
    preds = model.predict(input_data)

    df["prediction"] = preds
    return df[["machineID", "datetime", "prediction"]].to_dict(orient="records")
