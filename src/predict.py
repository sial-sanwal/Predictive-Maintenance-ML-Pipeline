import pandas as pd
import joblib
import os
from src.preprocess import preprocess_data_for_prediction


def predict():
    # Load data
    telemetry = pd.read_csv("test_telemetry.csv", parse_dates=["datetime"])
    machines = pd.read_csv("test_machines.csv")

    # Preprocess
    X = preprocess_data_for_prediction(telemetry, machines)

    # Preserve original machineID and datetime for output
    original_cols = X[["machineID", "datetime"]] if "datetime" in X.columns else X[["machineID"]]
    
    # Drop datetime if it's not needed for prediction
    X = X.drop(columns=["datetime"], errors="ignore")

    # Load trained model
    model = joblib.load("model.pkl")
    preds = model.predict(X)

    # Output results to console
    for i, pred in enumerate(preds):
        print(f"Machine ID: {original_cols.iloc[i]['machineID']} - Prediction: {pred}")

    # Save predictions to file
    output = original_cols.copy()
    output["prediction"] = preds

    os.makedirs("data/output", exist_ok=True)
    output.to_csv("data/output/predictions.csv", index=False)
    print("\n✅ Predictions saved to: data/output/predictions.csv")

if __name__ == "__main__":
    predict()
