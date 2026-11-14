from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
import app.main as api


class FakeModel:
    def predict(self, frame):
        assert list(frame.columns) == api.FEATURES
        return [2.5]


@pytest.fixture()
def client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    monkeypatch.setattr(api, "load_metadata", lambda: {
        "best_model": "Test Random Forest",
        "metrics": {"Test Random Forest": {"r2": 0.8}},
        "feature_importance": [{"feature": "MedInc", "importance": 50.0}],
    })
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def sample():
    return {
        "med_inc": 8.33,
        "house_age": 41,
        "ave_rooms": 6.98,
        "ave_bedrms": 1.02,
        "population": 322,
        "ave_occup": 2.56,
        "latitude": 37.88,
        "longitude": -122.23,
    }


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["docs"] == "/docs"


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["api_ready"] is True


def test_prediction_returns_usd_and_persists(client):
    response = client.post("/api/predict", json=sample())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price_usd"] == 250000
    assert body["lower_estimate_usd"] == 225000
    assert body["upper_estimate_usd"] == 275000
    assert body["inputs"]["latitude"] == 37.88
    assert client.get("/api/predictions").json()[0]["id"] == body["id"]


def test_rejects_invalid_california_coordinates(client):
    data = sample()
    data["latitude"] = 45.2
    assert client.post("/api/predict", json=data).status_code == 422


def test_rejects_fractional_population(client):
    data = sample()
    data["population"] = 322.5
    assert client.post("/api/predict", json=data).status_code == 422


def test_analytics_uses_saved_predictions(client):
    client.post("/api/predict", json=sample())
    body = client.get("/api/analytics").json()
    assert body["total_predictions"] == 1
    assert body["average_price"] == 250000
    assert sum(item["count"] for item in body["price_distribution"]) == 1
    assert body["best_model"] == "Test Random Forest"


def test_prediction_limit_validation(client):
    assert client.get("/api/predictions?limit=0").status_code == 422
