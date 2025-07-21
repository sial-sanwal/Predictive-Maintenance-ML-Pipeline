from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    """
    Builds and returns a machine learning pipeline consisting of a scaler and RandomForestClassifier.
    You can later replace RandomForestClassifier with other models without changing rest of the pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    return pipeline
