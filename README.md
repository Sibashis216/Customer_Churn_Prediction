# Customer Churn Prediction — Full-Stack App

Full frontend + backend built on top of your `Customer_Churn_Prediction_using_ML` notebook
(IBM Telco Customer Churn dataset, Random Forest classifier trained on SMOTE-balanced data).

```
churn-app/
├── backend/
│   ├── app.py                       # Flask API
│   ├── train_model.py               # Reproduces the notebook pipeline, saves .pkl artifacts
│   ├── requirements.txt             # Pinned dependency versions
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── customer_churn_model.pkl     # trained Random Forest + feature order
│   └── encoders.pkl                 # LabelEncoders for categorical fields
└── frontend/
    ├── index.html                   # "Churn Diagnostic Console" UI
    ├── style.css
    └── script.js
```

## 1. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python3 app.py
```

Runs on `http://127.0.0.1:5000`. Endpoints:

| Method | Path            | Purpose                                              |
|--------|-----------------|-------------------------------------------------------|
| GET    | `/api/health`   | Liveness check, used by the frontend status chip     |
| GET    | `/api/schema`   | Feature order + valid categorical values (for forms) |
| POST   | `/api/predict`  | Body = subscriber fields → churn prediction JSON     |

`/api/predict` request body must include all 19 model features exactly as named in the
original dataframe (`gender`, `SeniorCitizen`, `Partner`, … `TotalCharges`). See
`backend/app.py`'s `FEATURE_NAMES` / `FIELD_OPTIONS` for the authoritative list — the
`/api/schema` endpoint exposes the same thing so the frontend never hardcodes it blindly.

### Retraining

If you ever upgrade scikit-learn / imbalanced-learn / xgboost, **re-run `train_model.py`**
before restarting the API. This is the same joblib/pickle version-mismatch issue you hit on
the rainfall and rock-vs-mine projects — the `.pkl` files must be regenerated with whatever
versions are pinned in `requirements.txt` for that environment.

Training needs a couple of extra libraries not required to *serve* predictions
(`imbalanced-learn`, `xgboost`) — install those from `requirements-train.txt`:

```bash
pip install -r requirements-train.txt
python3 train_model.py
```

`requirements.txt` itself is deliberately slimmed down to only what `app.py` needs at
runtime, since that's the file Vercel bundles into the deployed function — see the
deployment section below.

## 5. Deploying free on Vercel

See `DEPLOYMENT.md` for the full click-by-click walkthrough.

## 2. Frontend

Pure HTML/CSS/JS, no build step. Two ways to run it:

- **Quick local check:** open `frontend/index.html` directly in a browser (works because
  `script.js` auto-detects `localhost` and points at `http://127.0.0.1:5000`).
- **Served properly:** `cd frontend && python3 -m http.server 8000`, then visit
  `http://localhost:8000`.

### Before deploying the frontend anywhere else (Vercel, Netlify, etc.)

Open `frontend/script.js` and replace the placeholder:

```js
// TODO: replace with your deployed backend URL
return "https://YOUR-BACKEND-DOMAIN.example.com";
```

with your actual deployed Flask API URL. This is the exact `ERR_CONNECTION_REFUSED` bug
you ran into on the Rock vs Mine project — a deployed frontend pointing at `localhost`
is calling the *visitor's* machine, not your server, so it must be updated to a real
backend origin before going live.

Also remember to enable CORS on the backend for your frontend's deployed origin (already
handled generically via `flask-cors` in `app.py`, but lock it down to your real domain in
production instead of the open default).

## 3. Design notes

The UI is styled as a telecom "diagnostic console" — a nod to the subject matter (a telco's
churn model): a radial signal-strength gauge shows churn probability, a rack of labeled
panels collects the subscriber's profile/services/billing, and a console log narrates each
API call. Colors and type (Space Grotesk / IBM Plex Mono / IBM Plex Sans) are loaded from
Google Fonts via CDN link tags in `index.html`.

## 4. Model summary

- Dataset: IBM Telco Customer Churn (7,043 rows, 19 features after dropping `customerID`)
- Class imbalance handled with SMOTE on the training split only
- Compared Decision Tree / Random Forest / XGBoost via 5-fold CV — Random Forest won
- Hold-out test accuracy: ~0.78 (see `train_model.py` output for the full classification report)
