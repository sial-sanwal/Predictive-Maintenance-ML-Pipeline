# tests/test_predict.py

import sys
import os

# Make sure root directory is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_post_predict():
    with open("test_telemetry.csv", "rb") as f1, open("test_machines.csv", "rb") as f2:
        response = client.post("/predict", files={
            "file_telemetry": ("test_telemetry.csv", f1, "text/csv"),
            "file_machines": ("test_machines.csv", f2, "text/csv")
        })

    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)
