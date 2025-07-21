import pandas as pd

# Load raw files
telemetry = pd.read_csv("data/raw/PdM_telemetry.csv", parse_dates=["datetime"])
machines = pd.read_csv("data/raw/PdM_machines.csv")

# Pick the last 3 days of telemetry (optional: choose specific machines)
test_telemetry = telemetry[telemetry["datetime"] >= telemetry["datetime"].max() - pd.Timedelta(days=3)]

# Save test data
test_telemetry.to_csv("test_telemetry.csv", index=False)
machines.to_csv("test_machines.csv", index=False)
