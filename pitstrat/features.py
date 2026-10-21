"""Turn cleaned laps into model inputs."""

import pandas as pd

# Air and track temperature were tried and dropped: they mostly identify a
# single race, so the model memorised races instead of learning tyre wear.
NUMERIC = ["TyreLife", "LapNumber", "RaceProgress"]
CATEGORICAL = ["Compound", "Circuit"]
TARGET = "LapDelta"


def make_features(laps, columns=None):
    """Return the feature matrix. Pass ``columns`` to match a trained model."""
    numeric = laps[["TyreLife", "LapNumber"]].astype(float)
    # Fuel burns off through the race, so progress stands in for car weight.
    numeric["RaceProgress"] = laps["LapNumber"] / laps["TotalLaps"]
    dummies = pd.get_dummies(laps[CATEGORICAL], dtype=float)
    features = pd.concat([numeric[NUMERIC], dummies], axis=1)
    if columns is not None:
        features = features.reindex(columns=columns, fill_value=0.0)
    return features
