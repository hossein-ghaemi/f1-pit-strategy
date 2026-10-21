"""Train Random Forest and XGBoost lap-time models and keep the better one."""

import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from pitstrat import config
from pitstrat.data import clean_laps
from pitstrat.features import TARGET, make_features


def candidate_models(columns):
    # An older tyre is never faster, so lap delta may only rise with tyre age.
    rising = [1 if column == "TyreLife" else 0 for column in columns]
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, monotonic_cst=rising, n_jobs=-1,
            random_state=0),
        "xgboost": XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=3, min_child_weight=100,
            reg_lambda=10, subsample=0.8, colsample_bytree=0.8,
            monotone_constraints=tuple(rising), random_state=0),
    }


def race_folds(laps, folds=5):
    """Split by whole races so every test fold holds races the model never saw."""
    races = laps["Year"].astype(str) + " " + laps["Circuit"]
    for train_index, test_index in GroupKFold(n_splits=folds).split(laps, groups=races):
        yield laps.iloc[train_index], laps.iloc[test_index]


def scores(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def train(raw_laps):
    """Cross-validate both models on held-out races, refit the best on all laps and save it."""
    laps = clean_laps(raw_laps)
    # Predicting every lap as the driver's median is the bar to beat.
    predictions = {"baseline": [], "random_forest": [], "xgboost": []}
    actual = []
    for train_laps, test_laps in race_folds(laps):
        X_train = make_features(train_laps)
        X_test = make_features(test_laps, X_train.columns)
        actual.append(test_laps[TARGET].to_numpy())
        predictions["baseline"].append(np.zeros(len(test_laps)))
        for name, model in candidate_models(X_train.columns).items():
            model.fit(X_train, train_laps[TARGET])
            predictions[name].append(model.predict(X_test))

    actual = np.concatenate(actual)
    results = {name: scores(actual, np.concatenate(pred)) for name, pred in predictions.items()}

    best = min(["random_forest", "xgboost"], key=lambda name: results[name]["mae"])
    X_all = make_features(laps)
    model = candidate_models(X_all.columns)[best].fit(X_all, laps[TARGET])

    bundle = {"name": best, "model": model, "columns": list(X_all.columns), "scores": results}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_FILE, "wb") as file:
        pickle.dump(bundle, file)
    return bundle


def load_model():
    with open(config.MODEL_FILE, "rb") as file:
        return pickle.load(file)


def predict(bundle, laps):
    return bundle["model"].predict(make_features(laps, bundle["columns"]))
