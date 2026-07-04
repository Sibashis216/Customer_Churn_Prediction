# Deploying "SIGNAL" Free on Vercel

You'll create **two separate Vercel projects** from the same GitHub repo:

1. `churn-backend` → deploys `backend/` as a Python serverless function (your Flask API)
2. `churn-frontend` → deploys `frontend/` as a static site (your HTML/CSS/JS console)

This is the same two-project pattern as your Rainfall and Rock-vs-Mine deployments —
Vercel doesn't run a long-lived Flask process, so backend and frontend are deployed and
scaled independently.

---

## Part A — Push the code to GitHub

1. Unzip `churn-app.zip` somewhere on your machine.
2. Go to https://github.com/new and create a new **empty** repository (no README, no
   .gitignore — you already have one), e.g. `Customer_Churn_Prediction` or reuse your
   existing repo.
3. In a terminal, from inside the unzipped `churn-app` folder:
   ```bash
   git init
   git add .
   git commit -m "Full-stack churn prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
4. Refresh the GitHub page — you should see `backend/`, `frontend/`, `README.md`, etc.

---

## Part B — Deploy the backend (Flask API)

1. Go to https://vercel.com and sign in (GitHub login is easiest — it's the free **Hobby** plan).
2. Click **Add New… → Project**.
3. Under "Import Git Repository," find your repo and click **Import**.
4. On the configuration screen:
   - **Project Name:** `churn-backend` (or anything you like — this becomes part of the URL)
   - **Root Directory:** click **Edit** → select `backend`. This is the most important step —
     it tells Vercel to treat `backend/` as its own project, using `backend/app.py` and
     `backend/requirements.txt`.
   - **Framework Preset:** Vercel should auto-detect "Flask" once the root directory is set
     to `backend`. If it shows "Other," that's fine too — the `vercel.json` inside `backend/`
     already tells Vercel how to run it.
   - Leave Build/Output/Install commands on their defaults (blank) — Vercel's Python runtime
     handles this automatically from `requirements.txt`.
5. Click **Deploy**. This takes 1–3 minutes — Vercel installs `flask`, `flask-cors`,
   `pandas`, `numpy`, `scikit-learn`, and bundles your two `.pkl` files.
6. When it finishes, click **Continue to Dashboard**, then **Visit**. You'll land on a URL
   like:
   ```
   https://churn-backend.vercel.app
   ```
   That URL is your live API base. Test it by appending `/api/health` in the browser:
   ```
   https://churn-backend.vercel.app/api/health
   ```
   You should see `{"status": "ok", "model": "RandomForestClassifier"}`. If you get an
   error instead, check **Deployments → (latest) → Functions → Logs** in the Vercel
   dashboard for the traceback.

**Copy that base URL** — you need it in Part C.

---

## Part C — Point the frontend at the deployed backend

Before deploying the frontend, tell it where the live API is:

1. Open `frontend/script.js` in a code editor.
2. Find this block near the top:
   ```js
   const API_BASE = (() => {
     if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
       return "http://127.0.0.1:5000";
     }
     // TODO: replace with your deployed backend URL
     return "https://YOUR-BACKEND-DOMAIN.example.com";
   })();
   ```
3. Replace the placeholder with your real backend URL from Part B (no trailing slash):
   ```js
     return "https://churn-backend.vercel.app";
   ```
4. Save, then commit and push:
   ```bash
   git add frontend/script.js
   git commit -m "Point frontend at deployed backend"
   git push
   ```

---

## Part D — Deploy the frontend (static site)

1. Back in Vercel: **Add New… → Project → Import** the **same** GitHub repo again.
2. This time:
   - **Project Name:** `churn-frontend`
   - **Root Directory:** click **Edit** → select `frontend`.
   - **Framework Preset:** "Other" is correct — it's plain HTML/CSS/JS, no build step needed.
3. Click **Deploy**.
4. When it finishes, click **Visit**. You'll get a URL like:
   ```
   https://churn-frontend.vercel.app
   ```
   This is your live app. Open it, fill in a subscriber profile, and click **Run
   diagnostic** — the status chip in the top-right should say "Model online," and the
   gauge should animate after you submit.

---

## Part E — Lock down CORS (optional but recommended)

Right now the backend accepts requests from any origin (`CORS(app)` in `app.py`), which is
fine for getting this working. Once you know your frontend's real Vercel URL, you can
restrict it:

1. Open `backend/app.py`, find:
   ```python
   CORS(app)  # allow the frontend (served separately / from file://) to call this API
   ```
2. Replace with:
   ```python
   CORS(app, origins=["https://churn-frontend.vercel.app"])
   ```
3. Commit and push — Vercel will auto-redeploy the backend project on the new push.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Frontend shows "API unreachable" | `API_BASE` in `script.js` still points at `localhost`, or wasn't pushed before the frontend deploy | Redo Part C, confirm the change is on GitHub, redeploy |
| Backend deploy fails with a size/build error | Extra training libraries (`xgboost`, `imbalanced-learn`) leaking into `backend/requirements.txt` | Make sure `requirements.txt` only has the 5 runtime packages (see `backend/requirements.txt`) — training-only deps belong in `requirements-train.txt` |
| `/api/predict` returns a CORS error in the browser console | Origin restricted in Part E doesn't exactly match your frontend URL | Double-check the exact `https://` URL (no trailing slash) in `CORS(app, origins=[...])` |
| Backend works when visited directly but frontend still fails | Browser cached the old `script.js` | Hard refresh (Ctrl/Cmd+Shift+R) or check the deployed file in Vercel's "Source" view |
| Pickle/version error in backend logs | Local `.pkl` files were trained with different library versions than `requirements.txt` pins | Re-run `pip install -r requirements-train.txt && python3 train_model.py` locally with those exact pinned versions, then re-push the regenerated `.pkl` files |

---

## Free-tier notes

- Both projects run on Vercel's **Hobby (free)** plan — no credit card required.
- Vercel Functions on Hobby have a **cold start** the first time they're hit after being
  idle; the first `/api/predict` call after a quiet period may take a couple of seconds
  longer while the model unpickles. Subsequent calls are fast.
- Every `git push` to `main` auto-redeploys both projects independently, since they're
  scoped to different root directories.
