from data_loader import load_raw_data
from preprocess import preprocess_and_merge
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_model():
    telemetry, machines, errors, maint, failures = load_raw_data()
    df = preprocess_and_merge(telemetry, machines, errors, maint, failures)

    X = df.drop(["datetime", "failure_within_24h"], axis=1)
    y = df["failure_within_24h"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
