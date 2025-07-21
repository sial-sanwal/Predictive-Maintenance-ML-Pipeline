import pandas as pd

def load_raw_data():
    telemetry = pd.read_csv("data/raw/PdM_telemetry.csv", parse_dates=["datetime"])
    machines = pd.read_csv("data/raw/PdM_machines.csv")
    errors = pd.read_csv("data/raw/PdM_errors.csv", parse_dates=["datetime"])
    maint = pd.read_csv("data/raw/PdM_maint.csv", parse_dates=["datetime"])
    failures = pd.read_csv("data/raw/PdM_failures.csv", parse_dates=["datetime"])
    return telemetry, machines, errors, maint, failures
