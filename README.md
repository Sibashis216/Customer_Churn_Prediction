# Customer Churn Prediction

> A full-stack machine learning application that predicts telecom customer churn and explains the drivers behind each prediction — built end-to-end from EDA through a deployed, interactive web app.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg?style=flat-square)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg?style=flat-square)](https://flask.palletsprojects.com/)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed-Vercel-black.svg?style=flat-square)](https://vercel.com)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-77.86%25-brightgreen.svg?style=flat-square)]()
[![AUC--ROC](https://img.shields.io/badge/AUC--ROC-0.8234-brightgreen.svg?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](LICENSE)

**🚀 [Live Demo](https://customer-churn-prediction-lorv.vercel.app/)** &nbsp;|&nbsp; **📓 [Notebook](Customer_Churn_Prediction_using_ML.ipynb)** &nbsp;|&nbsp; **🐛 [Report an Issue](../../issues)**

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Key EDA Findings](#key-eda-findings)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Limitations & Future Work](#limitations--future-work)
- [Author](#author)
- [License](#license)

---

## Overview

Telecom providers lose a meaningful share of subscribers every year, and by the time a customer actually cancels, it's too late to intervene. This project builds a model that flags at-risk customers *before* they churn, using their profile, service usage, and billing history — and wraps it in a usable interface rather than leaving it as a notebook.

**What it does:**
- Predicts churn probability for a subscriber from 19 profile, service, and billing features
- Surfaces the top features driving each individual prediction, so the output is explainable rather than a black-box score
- Serves predictions through a REST API and a custom-built frontend styled as a telecom diagnostic console

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features, 26.54% churn rate.

## Live Demo

**[customer-churn-prediction-lorv.vercel.app](https://customer-churn-prediction-lorv.vercel.app/)**

Fill in a subscriber's profile, service lines, and billing details, then run the diagnostic to see the churn probability animate on a live gauge, along with the top drivers behind that specific prediction.

## Key EDA Findings

All figures below are computed directly from the dataset (`n = 7,043`).

### Contract type is the single strongest churn signal

| Contract Type | Churn Rate | Customers |
|---|---|---|
| Month-to-month | **42.71%** | 3,875 |
| One year | 11.27% | 1,473 |
| Two year | 2.83% | 1,695 |

Month-to-month subscribers churn at roughly **15x** the rate of two-year contract holders.

### Fiber optic customers churn disproportionately

| Internet Service | Churn Rate | Customers |
|---|---|---|
| Fiber optic | **41.89%** | 3,096 |
| DSL | 18.96% | 2,421 |
| No internet | 7.40% | 1,526 |

### Senior citizens churn at nearly double the general rate

| Segment | Churn Rate | Customers |
|---|---|---|
| Senior citizens | **41.68%** | 1,142 |
| Non-senior | 23.61% | 5,901 |

### Tenure and pricing

- Churned customers: average tenure **17.98 months** (median 10) vs. **37.57 months** (median 38) for retained customers — the first ~18 months is the highest-risk window.
- Churned customers pay **$74.44/month** on average vs. **$61.27/month** for retained customers.
- Customers **without online security** churn at 41.77% vs. 14.61% for those with it — a larger gap than tech support (which has a much smaller effect).

## Model Performance

Three classifiers were compared via 5-fold cross-validation on SMOTE-balanced training data (to correct for the 73%/27% class imbalance):

| Algorithm | 5-Fold CV Accuracy |
|---|---|
| Decision Tree | 80.25% |
| **Random Forest** ✅ | **84.08%** |
| XGBoost | 83.13% |

**Random Forest** was selected as the final model. Held-out test set performance (20% split, never seen during training or SMOTE resampling):

| Metric | Score |
|---|---|
| Accuracy | 77.86% |
| AUC-ROC | 0.8234 |
| Precision (Churn class) | 58.09% |
| Recall (Churn class) | 58.71% |
| F1-Score (Churn class) | 0.58 |

```
Confusion Matrix
                 Predicted: No Churn   Predicted: Churn
Actual: No Churn        878                  158
Actual: Churn           154                  219
```

The model correctly identifies 85% of customers who stay, and catches roughly 59% of customers who actually churn — reasonable for a real-world imbalanced problem, but not perfect (see [Limitations](#limitations--future-work)).

## Feature Importance

Top 5 features by Random Forest importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Total Charges | 0.1416 |
| 2 | Monthly Charges | 0.1365 |
| 3 | Contract | 0.1266 |
| 4 | Tenure | 0.1218 |
| 5 | Online Security | 0.0867 |

Billing history, contract commitment, and tenure dominate — consistent with the EDA findings above.

## Tech Stack

**Machine Learning**
- Python, pandas, NumPy
- scikit-learn (Random Forest, Decision Tree)
- XGBoost (comparison baseline)
- imbalanced-learn (SMOTE)

**Backend**
- Flask + Flask-CORS
- REST API: `/api/predict`, `/api/schema`, `/api/health`
- Model persisted with `pickle`

**Frontend**
- Vanilla HTML5 / CSS3 / JavaScript (no framework)
- Custom-animated SVG gauge, no charting library

**Deployment**
- Vercel — backend as a Python serverless function, frontend as a static site (two independent projects from one repo)

## Project Structure

```
Customer_churn_Prediction/
├── Customer_Churn_Prediction_using_ML.ipynb   # EDA + model development notebook
├── backend/
│   ├── app.py                                 # Flask API
│   ├── train_model.py                         # Reproduces the notebook pipeline
│   ├── customer_churn_model.pkl                # Trained Random Forest + feature order
│   ├── encoders.pkl                            # LabelEncoders for categorical fields
│   ├── requirements.txt                        # Runtime-only dependencies (deploy)
│   ├── requirements-train.txt                  # Full dependencies (local retraining)
│   ├── vercel.json                             # Vercel function config
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── frontend/
│   ├── index.html                              # Diagnostic console UI
│   ├── style.css
│   └── script.js
├── DEPLOYMENT.md                               # Step-by-step Vercel deployment guide
└── README.md
```

## Setup & Installation

### Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Runs on `http://127.0.0.1:5000`. Verify with:

```bash
curl http://127.0.0.1:5000/api/health
```

### Frontend

```bash
cd frontend
python3 -m http.server 8000
```

Open `http://localhost:8000`. `script.js` auto-detects `localhost` and points at the local backend — no configuration needed for local development.

### Retraining the model

```bash
cd backend
pip install -r requirements-train.txt
python3 train_model.py
```

Re-run this whenever `scikit-learn` / `imbalanced-learn` / `xgboost` versions change, so the pickled model matches the environment that loads it.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/schema` | Feature order + valid categorical values |
| `POST` | `/api/predict` | Returns prediction, churn probability, and top feature drivers |

## Deployment

Deployed as two independent Vercel projects from this single repository:

1. **Backend** — root directory `backend/`, deployed as a Python serverless function
2. **Frontend** — root directory `frontend/`, deployed as a static site, pointed at the backend's URL

Full click-by-click steps, including CORS configuration and troubleshooting, are in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Limitations & Future Work

Being upfront about what this model doesn't do well, and what's still on the roadmap:

- **Recall on churners is ~59%** — the model misses roughly 4 in 10 customers who actually churn. For a production retention system, this would need to be weighed against the cost of false positives (offering discounts to customers who weren't going to leave anyway).
- **No real-time data pipeline** — predictions are one-off API calls; there's no batch scoring or scheduled retraining job yet.
- **Planned improvements:**
  - Hyperparameter tuning (grid/random search) on the Random Forest
  - Threshold tuning based on a defined cost matrix instead of the default 0.5 cutoff
  - Batch prediction endpoint for scoring a full customer list at once
  - Model monitoring for data/prediction drift over time

## Author

**Sibashis Patnaik**

- 📧 Email: [sibashispatnaik8@gmail.com](mailto:sibashispatnaik8@gmail.com)
- 🔗 LinkedIn: [Sibashis Patnaik](https://www.linkedin.com/in/sibashis-patnaik-ai/)
- 🐙 GitHub: [@Sibashis216](https://github.com/Sibashis216)
- 💼 Portfolio: [sibashis-patnaik-portfol-vx7gfr3.gamma.site](https://sibashis-patnaik-portfol-vx7gfr3.gamma.site/)

## License

Licensed under the [MIT License](LICENSE).
