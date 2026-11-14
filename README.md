# EstatePredict AI

EstatePredict is a full-stack educational machine-learning application. It estimates historical median house values for California census block groups using the California Housing dataset. It is not a current individual-home appraisal or live property valuation service.

## Features

- Home, Predict Price, Prediction Result, Analytics and About views.
- Eight validated property/neighborhood inputs and a working sample-data action.
- FastAPI prediction endpoint using the selected trained model.
- SQLite persistence for successful predictions.
- Analytics for saved predictions, average/median estimate, distribution and model feature importance.
- Model evaluation metrics for Linear Regression and Random Forest.
- Health endpoint reporting API, database and model readiness.
- Helpful API errors instead of attempting to parse empty or invalid responses as JSON.
- Responsive desktop/mobile navigation and layouts.

## Important dataset limitation

The California Housing dataset contains 20,640 historical census block-group observations. Its target is in units of $100,000 and is converted to USD by the API. It does not contain live listings, current market conditions, property condition or renovation details. The displayed ±10% band is illustrative only and is not a calibrated confidence interval.

## Run on macOS

The project needs Python 3.10+ and Node.js 20+. On this Mac, use `python3.11` explicitly because the system `python3` may point to an older version.

### 1. Backend (Terminal 1)

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.ml.train
uvicorn app.main:app --reload
```

Keep this terminal open. The first training run downloads the California Housing dataset. If Python reports an SSL certificate error on macOS, run `/Applications/Python\\ 3.11/Install\\ Certificates.command` once, then rerun the training command.

### 2. Frontend (Terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Keep this terminal open and visit http://localhost:5173. Vite proxies `/api` requests to the backend at http://127.0.0.1:8000.

### 3. Verify the services

- Open http://127.0.0.1:8000/api/health. `api_ready`, `database_ready` and `model_ready` should all be `true`.
- Open http://127.0.0.1:8000/docs to inspect and exercise the API.
- In the app, open Predict Price, use sample data, and submit. A result should appear and be saved.
- Open Analytics. Total predictions should increase and the recent-predictions list and distribution should update.

## Run tests

From the `backend` folder with the virtual environment active:

```bash
python -m pytest -q
```

## API routes

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | API, database and model readiness |
| GET | `/api/model` | Model metadata and evaluation metrics |
| POST | `/api/predict` | Validate, estimate and persist an input record |
| GET | `/api/predictions` | List recent saved predictions (`limit` 1–100) |
| GET | `/api/analytics` | Prediction aggregates, distribution and feature importance |

## Data and privacy

Inputs are saved locally in the SQLite database so the Analytics page can summarize previous predictions. This starter project has no accounts or multi-user access control. Do not expose it publicly with sensitive data without adding authentication, authorization, HTTPS, rate limiting, backups and operational monitoring.
