from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Prediction
from .schemas import PropertyInput

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "best_model.joblib"
META_PATH = ARTIFACTS / "metadata.json"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

app = FastAPI(
    title="EstatePredict AI API",
    version="1.1.0",
    description="Educational estimates from historical California Housing census data.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {
        "service": "EstatePredict AI API",
        "docs": "/docs",
        "health": "/api/health",
        "api_ready": True,
    }


def load_metadata() -> dict[str, Any] | None:
    if not META_PATH.exists():
        return None
    try:
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Model metadata is missing or damaged. Retrain the model with: python -m app.ml.train") from exc
    return data if isinstance(data, dict) else None


_MODEL_CACHE = None
_MODEL_MTIME: int | None = None


def load_model():
    global _MODEL_CACHE, _MODEL_MTIME
    if not MODEL_PATH.is_file():
        return None
    modified = MODEL_PATH.stat().st_mtime_ns
    if _MODEL_CACHE is None or _MODEL_MTIME != modified:
        try:
            _MODEL_CACHE = joblib.load(MODEL_PATH)
            _MODEL_MTIME = modified
        except Exception as exc:
            _MODEL_CACHE = None
            _MODEL_MTIME = None
            raise HTTPException(status_code=503, detail="The trained model file could not be loaded. Retrain it with: python -m app.ml.train") from exc
    return _MODEL_CACHE


def require_model() -> tuple[Any, dict[str, Any]]:
    model = load_model()
    metadata = load_metadata()
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="The ML model is not ready. From the backend folder, run: python -m app.ml.train")
    return model, metadata


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    database_ready = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database_ready = False
    model_ready = MODEL_PATH.is_file() and META_PATH.is_file()
    return {
        "status": "ok" if database_ready else "degraded",
        "api_ready": True,
        "database_ready": database_ready,
        "model_ready": model_ready,
    }


@app.get("/api/model")
def model_info():
    _, metadata = require_model()
    return metadata


@app.post("/api/predict")
def predict(payload: PropertyInput, db: Session = Depends(get_db)):
    model, metadata = require_model()
    values = [payload.med_inc, payload.house_age, payload.ave_rooms, payload.ave_bedrms,
              payload.population, payload.ave_occup, payload.latitude, payload.longitude]
    row = pd.DataFrame([values], columns=FEATURES)
    try:
        raw_prediction = float(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model could not calculate this estimate. Check the backend logs.") from exc
    estimate = max(0.0, raw_prediction * 100000.0)
    record = Prediction(
        model_name=str(metadata.get("best_model", "Unknown")),
        predicted_price_usd=estimate,
        med_inc=payload.med_inc, house_age=payload.house_age, ave_rooms=payload.ave_rooms,
        ave_bedrms=payload.ave_bedrms, population=payload.population, ave_occup=payload.ave_occup,
        latitude=payload.latitude, longitude=payload.longitude,
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="The estimate was calculated, but it could not be saved to the database.") from exc

    return {
        "id": record.id,
        "predicted_price_usd": round(estimate, 2),
        "lower_estimate_usd": round(estimate * 0.9, 2),
        "upper_estimate_usd": round(estimate * 1.1, 2),
        "model_name": record.model_name,
        "created_at": record.created_at.isoformat() if record.created_at else datetime.now(timezone.utc).isoformat(),
        "inputs": payload.model_dump(),
        "disclaimer": "Illustrative ±10% range only; it is not a calibrated confidence interval. This is a historical census estimate, not a current individual-home appraisal.",
    }


@app.get("/api/predictions")
def recent_predictions(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit).all()
    return [{
        "id": row.id,
        "predicted_price_usd": row.predicted_price_usd,
        "model_name": row.model_name,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "med_inc": row.med_inc,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    } for row in rows]


@app.get("/api/analytics")
def analytics(db: Session = Depends(get_db)):
    _, metadata = require_model()
    rows = db.query(Prediction).all()
    prices = [float(row.predicted_price_usd) for row in rows]
    metric_sets = metadata.get("metrics", {})
    best_model = metadata.get("best_model")
    metrics = metric_sets.get(best_model, {})
    edges = [0, 100000, 200000, 300000, 400000, 500000, float("inf")]
    labels = ["<100k", "100–200k", "200–300k", "300–400k", "400–500k", "500k+"]
    counts = [0] * len(labels)
    for price in prices:
        for index in range(len(labels)):
            if edges[index] <= price < edges[index + 1]:
                counts[index] += 1
                break
    return {
        "total_predictions": len(prices),
        "average_price": round(float(np.mean(prices)), 2) if prices else None,
        "median_price": round(float(np.median(prices)), 2) if prices else None,
        "r2": metrics.get("r2"),
        "price_distribution": [{"range": label, "count": count} for label, count in zip(labels, counts)],
        "feature_importance": metadata.get("feature_importance", []),
        "metrics": metric_sets,
        "model_metrics": metric_sets,
        "best_model": best_model,
    }
