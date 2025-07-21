import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

def preprocess_and_merge(telemetry, machines, errors, maint, failures):
    # Step 1: Aggregate telemetry (e.g., 3-hour averages)
    agg_telemetry = telemetry.groupby(
        ['machineID', pd.Grouper(key='datetime', freq='3h')]  
    ).mean().reset_index()

    # Step 2: Merge with machine metadata
    df = agg_telemetry.merge(machines, on='machineID', how='left')

    # Step 3: Encode the 'model' column using LabelEncoder
    le = LabelEncoder()
    df['model'] = le.fit_transform(df['model'])

    # Step 4: Add binary target label - failure within 24 hours
    failures['label'] = 1
    df['failure_within_24h'] = 0
    for i, row in failures.iterrows():
        mid = row['machineID']
        fail_time = row['datetime']
        mask = (
            (df['machineID'] == mid) &
            (df['datetime'] >= fail_time - timedelta(hours=24)) &
            (df['datetime'] < fail_time)
        )
        df.loc[mask, 'failure_within_24h'] = 1

    # Step 5: Drop any NaNs created from rolling or merging
    return df.dropna()

def preprocess_data_for_prediction(telemetry, machines):

    agg_telemetry = telemetry.groupby(
        ['machineID', pd.Grouper(key='datetime', freq='3h')]
    ).mean().reset_index()

    # Merge with machine metadata
    df = agg_telemetry.merge(machines, on='machineID', how='left')

    # Encode the 'model' column
    le = LabelEncoder()
    df['model'] = le.fit_transform(df['model'])

    return df.dropna()
