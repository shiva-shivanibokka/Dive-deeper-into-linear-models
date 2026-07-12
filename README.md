# Dive Deeper into Linear Models

> An interactive playground + notebook series exploring 20+ regression and classification models on real datasets — deployed live on Vercel.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![Deployed on Vercel](https://img.shields.io/badge/deployed-Vercel-000)

**🔗 Live demo: [web-sable-gamma-39.vercel.app](https://web-sable-gamma-39.vercel.app)**

---

## Recruiter TL;DR

- **What it is:** two teaching notebooks covering 20+ supervised-learning models (regularized, kernel, robust, generalized, and probabilistic linear models plus tree ensembles), paired with a **16-tab interactive web app** that lets anyone explore each model by dragging sliders and watching charts redraw — deployed live on Vercel.
- **Hardest problem solved well:** making trained scikit-learn models *interactive in the browser with no backend and no ML runtime client-side* — models are fit once in Python, swept over their key parameter, and exported as compact precomputed JSON that a dependency-free React/SVG layer redraws instantly.
- **Why it matters:** closes the most common student-portfolio gap — a model that "works in a notebook but was never deployed." Here the models are deployed, and made legible to a non-expert.

---

## Overview

Linear and kernel-based models are the backbone of applied machine learning, but most learning resources stop at a static plot. This project turns the "dive deeper" material into something you can *poke at*: move a regularization penalty and watch Lasso zero out features in real time; inject outliers and watch ordinary least squares swing while robust regressors hold; drag a decision threshold and watch precision trade against recall on a live ROC curve.

It was built as both a teaching resource and a portfolio piece — the notebooks are the reference, and the web app is the proof that the models were deployed and communicated, not just trained.

**Scope note (no redundancy):** plain linear and logistic regression *fundamentals* (pipelines, leakage-free preprocessing, honest evaluation) live in the companion repo **[Basics of Linear & Logistic Regression](https://github.com/shiva-shivanibokka/Basics-of-Linear-and-Logistic-Regression)**. This repo deliberately goes deeper into the regularized, kernel, robust, generalized, and probabilistic members of the family.

---

## The Interactive Playground

Sixteen demo tabs, each reading one precomputed artifact and rendering it as inline SVG:

| # | Tab | What you do | Models | Dataset |
|---|-----|-------------|--------|---------|
| 1 | Linear Regression | pick a feature → best-fit line, R², RMSE | OLS | California Housing |
| 2 | Bias–Variance | drag polynomial degree 1→15 → watch over/underfitting | Polynomial | Diabetes |
| 3 | Regularization | slide α → coefficient paths shrink, Lasso zeroes out | Ridge, Lasso, ElasticNet | Diabetes |
| 4 | KNN & SVR | change k / ε-tube → fit reshapes | KNN, SVR | California Housing |
| 5 | Bayesian Regression | raise prior strength → uncertainty band widens | Bayesian linear | Diabetes |
| 6 | Robust Regression | inject outliers → OLS swings, robust fits hold | Huber, RANSAC, Theil-Sen | Diabetes |
| 7 | Generalized Linear Models | toggle GLM → OLS predicts negatives, GLMs don't | Poisson, Gamma, Tweedie | California Housing |
| 8 | Quantile Regression | slide quantile 0.1→0.9 → build a prediction interval | Quantile | Diabetes |
| 9 | Discriminant Analysis | toggle LDA/QDA → linear vs curved boundary + ellipses | LDA, QDA | Wine |
| 10 | Naive Bayes | see class-conditional decision regions | Gaussian NB | Iris |
| 11 | KNN & SVM | change k / kernel → boundary reshapes | KNN, SVM | Iris |
| 12 | Linear Classifiers | step Perceptron iterations → boundary settles | Perceptron, Ridge Classifier | Iris |
| 13 | Boosting & Importance | compare importances + partial dependence | Gradient Boosting, XGBoost | Breast Cancer |
| 14 | Threshold, ROC & Metrics | drag threshold → confusion matrix + ROC/PR move live | Gradient Boosting | Breast Cancer |
| 15 | Model Comparison | sortable leaderboard across every model | all | Diabetes + Breast Cancer |
| 16 | About | plain-language guide to every model | — | — |

---

## Architecture

The core decision: **precompute in Python, interpolate in the browser.** No model runs client-side and there is no backend — every demo reads a small JSON artifact that was computed once by sweeping a model's key parameter offline. This keeps the site fully static (instant, free to host, nothing to secure) while still feeling live.

```mermaid
flowchart LR
    subgraph Python["Python (offline, run once)"]
        DS[5 real sklearn datasets] --> EX[export_web_artifacts.py]
        NB[Jupyter notebooks<br/>20+ model sections] -.teaches.-> EX
        EX -->|sweeps each model's<br/>key parameter| JSON[web/public/*.json<br/>15 precomputed artifacts]
    end
    subgraph Browser["Next.js on Vercel (static, client-side)"]
        JSON --> HOOK[useArtifact fetch hook]
        HOOK --> TAB[16 lazy-loaded tab components]
        TAB --> SVG[dependency-free inline-SVG charts]
        CTRL[slider / toggle] -->|indexes into<br/>precomputed arrays| TAB
    end
```

**Why this shape:**
- *Precomputed artifacts over live inference* — porting sklearn models (SVR, Bayesian ridge, XGBoost) to JS/WASM would be far more work and fragile; precomputing gives the same interactivity for a fraction of the effort and ships zero ML runtime to the client.
- *Inline SVG over a chart library* — the whole frontend depends only on `next`, `react`, and `react-dom`. Every chart (scatter, coefficient paths, decision-boundary heat maps, ROC curves, covariance ellipses) is drawn by a small shared SVG helper module.
- *Committed artifacts* — the JSON is committed, so the Vercel build needs no Python; it just builds the Next app.

---

## Tech Stack

| Layer | Choice | Version | Why |
|-------|--------|---------|-----|
| Models & data | scikit-learn, XGBoost, NumPy | sklearn 1.8, xgb 3.2 | every model here is standard sklearn/xgboost; datasets ship with sklearn (no downloads) |
| Export | Python | 3.12 | one script, one builder per artifact, with an assert-based self-check |
| Frontend | Next.js 15 · React 19 · TypeScript | Next 15, React 19 | App Router + lazy tab loading; strict TS |
| Charts | inline SVG | — | zero chart-library dependencies |
| Hosting | Vercel | — | static client-side app, auto-detected Next.js |

---

## Skills Demonstrated

- **Applied ML breadth** — regularization, robust regression, generalized linear models, discriminant analysis, kernel methods, and boosting, applied and compared on real datasets.
- **Cloud deployment (Vercel)** — a working, publicly reachable production URL, not a local-only demo.
- **System design & architecture** — a documented, deliberate tradeoff (precomputed artifacts vs. client-side inference) recorded in `docs/superpowers/specs/`.
- **Data-to-artifact pipeline** — a reproducible export stage that transforms raw sklearn datasets into model-ready, versioned JSON with a validation self-check.
- **Frontend engineering** — TypeScript + React/Next.js, dependency-light, accessible controls, responsive SVG.
- **Technical communication** — turning ML concepts into interactive explanations a non-expert can follow.

---

## Project Structure

```
Dive-deeper-into-linear-models/
├── linear_regression_models.ipynb     # 11 regression model sections (incl. Robust, GLM, Quantile)
├── classification_models.ipynb        # 7 classification model sections
├── requirements.txt                   # Python deps for the notebooks + export
├── scripts/
│   └── export_web_artifacts.py        # sklearn models → web/public/*.json (with self-check)
├── web/                               # Next.js 15 interactive playground
│   ├── app/
│   │   ├── page.tsx                   # tab shell (hero, tab bar, panel)
│   │   ├── models.ts                  # tab registry — single source of truth
│   │   ├── lib/                       # useArtifact hook + shared SVG chart helpers
│   │   └── components/                # 16 tab components (one per demo)
│   └── public/                        # 15 precomputed JSON artifacts
└── docs/superpowers/                  # design spec + implementation plan
```

---

## Getting Started

### Run the notebooks

```bash
git clone https://github.com/shiva-shivanibokka/Dive-deeper-into-linear-models.git
cd Dive-deeper-into-linear-models
pip install -r requirements.txt
jupyter notebook          # open either .ipynb and run top-to-bottom
```

The notebooks also run in [Google Colab](https://colab.research.google.com/) with no local setup.

### Run the web app locally

```bash
cd web
npm install
npm run dev               # http://localhost:3000
```

### Regenerate the precomputed artifacts

Only needed if you change the data or models:

```bash
python scripts/export_web_artifacts.py   # writes web/public/*.json, then self-checks
```

---

## Deployment

Deployed on **Vercel** as a static, client-side Next.js app (project root: `web/`). The site is fully static — no server functions, no environment variables, no Python at build time (the precomputed artifacts are committed). Live at **[web-sable-gamma-39.vercel.app](https://web-sable-gamma-39.vercel.app)**.

> Note: the Vercel project's **Root Directory** must be set to `web` for git-triggered redeploys, since the Next.js app lives in a subdirectory.

---

## Testing

There is no formal unit-test suite (this is a teaching/portfolio repo, not a service). What *is* verified:

- `export_web_artifacts.py` ends with an **assert-based self-check** that every expected artifact exists, is valid JSON, and is non-empty — a failed or empty export is caught before it can ship.
- Both notebooks execute **top-to-bottom without errors** (verified via `jupyter nbconvert --execute`).
- The web app builds clean with **strict TypeScript** (`next build` with no type errors).

---

## Roadmap / Known Limitations

- Demos show a small, representative parameter sweep per model (e.g. a handful of α values), not a continuous fit — a deliberate size/latency tradeoff of the precomputed approach.
- Bias–Variance, Bayesian, Robust, and Quantile demos use a single real feature to keep the curve visualizable in 2D.
- Possible next steps: a continuous-parameter mode via lightweight client-side interpolation, and per-tab links back to the exact notebook cell.

---

## License

MIT — see [LICENSE](./LICENSE).

---

## Author

**Shivani Bokka** — built as a teaching resource and portfolio piece demonstrating applied supervised learning, front-end engineering, and cloud deployment end to end.
