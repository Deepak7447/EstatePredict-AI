from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae_usd": round(float(mean_absolute_error(y_true, y_pred) * 100000), 2),
        "mse_usd2": round(float(mse * 100000**2), 2),
        "rmse_usd": round(float(np.sqrt(mse) * 100000), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dataset = fetch_california_housing(as_frame=True)
    X, y = dataset.data[FEATURES], dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=220, max_depth=24, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            )),
        ]),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(y_test, model.predict(X_test))
    best_name = max(results, key=lambda n: results[n]["r2"])
    joblib.dump(models[best_name], ARTIFACTS / "best_model.joblib")
    joblib.dump(models, ARTIFACTS / "models.joblib")
    metadata = {
        "best_model": best_name,
        "features": FEATURES,
        "dataset_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": results,
        "feature_importance": _feature_importance(models[best_name], FEATURES),
        "target_note": "California Housing target is in units of $100,000; display values are converted to USD.",
        "dataset_note": "Historical California census block-group data; not current individual-home appraisal data."
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

def _feature_importance(pipeline, feature_names):
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        vals = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        vals = np.abs(estimator.coef_)
    else:
        return []
    total = float(np.sum(vals)) or 1.0
    return [
        {"feature": name, "importance": round(float(value / total * 100), 2)}
        for name, value in sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
    ]

if __name__ == "__main__":
    main()
