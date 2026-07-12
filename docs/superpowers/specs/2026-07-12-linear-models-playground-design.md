# Linear Models Playground — Design Spec

**Date:** 2026-07-12
**Author:** Shivani Bokka (with Claude Code)
**Repo:** `Dive-deeper-into-linear-models`

## Goal

Bring this educational repo to parity with the sibling `Dive-Deeper-into-Supervised-Learning`
repo: add a deployed, interactive Next.js frontend on Vercel that lets visitors explore every
linear/kernel model in the two notebooks by moving sliders and watching charts redraw. Also
close the basic repo-polish gaps (requirements, license, gitignore) and refresh the README last.

This is explicitly a **portfolio / resume** piece. The win is a live URL that proves the models
were deployed and made legible, not just left in a notebook.

## Cross-repo scope (avoid redundancy)

Plain **Linear Regression** and **Logistic Regression** fundamentals (pipelines, leakage-free
preprocessing, evaluation) are already covered in the sibling repo
`Basics-of-Linear-and-Logistic-Regression` (diamonds dataset). This repo deliberately does NOT
duplicate them — it goes "deeper" into the regularized, kernel, probabilistic, robust, and
generalized members of the linear-model family, plus a capstone comparison.

## Non-goals

- No live ML inference in the browser. All demos read **precomputed JSON grids** exported from
  Python. The browser interpolates/redraws; it never trains or predicts.
- No synthetic datasets. Every demo uses a real `sklearn.datasets` dataset.
- No chart libraries. All visuals are inline SVG, matching the sample repo (zero deps beyond
  next/react).
- No backend, no database, no auth. Fully static client-side site.

## Architecture

```
Dive-deeper-into-linear-models/
├── linear_regression_models.ipynb     (unchanged)
├── classification_models.ipynb        (unchanged)
├── requirements.txt                    NEW  pinned deps for notebooks + export
├── LICENSE                             NEW  MIT
├── .gitignore                          NEW  Python + Next.js
├── scripts/
│   └── export_web_artifacts.py         NEW  sklearn -> web/public/*.json
├── web/                                NEW  Next.js 15 app (copied structure from sample)
│   ├── package.json                    next, react, react-dom only
│   ├── next.config.mjs
│   ├── tsconfig.json
│   ├── app/
│   │   ├── page.tsx                    tab shell (hero, tab bar, panel)
│   │   ├── models.ts                   MODEL_TABS registry (single source of truth)
│   │   ├── globals.css                 theme, layout, light/dark
│   │   ├── lib/
│   │   │   ├── svg.tsx                  tiny SVG chart helpers (axes, line, scatter, heat)
│   │   │   └── tip.tsx                  "?" tooltip (from sample)
│   │   └── components/
│   │       ├── LinearTab.tsx
│   │       ├── BiasVarianceTab.tsx
│   │       ├── RegularizationTab.tsx
│   │       ├── KnnSvrTab.tsx
│   │       ├── BayesianTab.tsx
│   │       ├── RobustTab.tsx
│   │       ├── GlmTab.tsx
│   │       ├── QuantileTab.tsx
│   │       ├── DiscriminantTab.tsx
│   │       ├── NaiveBayesTab.tsx
│   │       ├── KnnSvmTab.tsx
│   │       ├── LinClfTab.tsx
│   │       ├── BoostingTab.tsx
│   │       ├── ThresholdTab.tsx
│   │       ├── ComparisonTab.tsx
│   │       └── AboutTab.tsx
│   └── public/
│       ├── linear.json
│       ├── biasvariance.json
│       ├── regularization.json
│       ├── knnsvr.json
│       ├── bayesian.json
│       ├── robust.json
│       ├── glm.json
│       ├── quantile.json
│       ├── discriminant.json
│       ├── naivebayes.json
│       ├── knnsvm.json
│       ├── linclf.json
│       ├── boosting.json
│       ├── threshold.json
│       └── comparison.json
```

**Data flow:** `export_web_artifacts.py` loads each real dataset, fits the relevant models over
a swept parameter range, and writes the results (coefficients, curves, boundary grids, metric
tables) as JSON into `web/public/`. Each tab `fetch()`es its one JSON file on mount, holds it in
state, and re-renders SVG when a control changes. The swept parameter's current value just indexes
into the precomputed arrays.

## Datasets (all `sklearn.datasets`, real, no download beyond sklearn's cache)

| Dataset | Loader | Used by |
|---|---|---|
| California Housing | `fetch_california_housing` | Linear, KNN&SVR |
| Diabetes | `load_diabetes` | Bias–Variance, Regularization, Bayesian |
| Iris | `load_iris` | Naive Bayes, KNN&SVM |
| Wine | `load_wine` | Discriminant |
| Breast Cancer | `load_breast_cancer` | Boosting, Threshold |

## Tabs & artifact contracts

Each tab is independent: one component, one JSON file, one clear concept. Artifact schemas below
are the contract between `export_web_artifacts.py` and the component.

1. **Linear Regression** — `linear.json` *(California Housing)*
   Pick 2 of 8 features → best-fit line on the stronger feature, residual scatter, R²/RMSE.
   Schema: `{ features: string[], perFeature: { [name]: { coef, r2, rmse, points:[[x,y]], fit:[[x,y]] } } }`

2. **Polynomial & Bias–Variance** — `biasvariance.json` *(Diabetes, BMI feature only)*
   Degree slider 1→15 → fitted curve over real BMI→progression scatter + train/test error lines.
   Schema: `{ scatter:[[x,y]], degrees:[{ degree, fit:[[x,y]], trainErr, testErr }] }`

3. **Regularization** — `regularization.json` *(Diabetes)*
   `alpha` slider (log grid) → Ridge/Lasso/ElasticNet coefficient paths; Lasso drives coefs to 0.
   Schema: `{ features:string[], alphas:number[], ridge:number[][], lasso:number[][], elastic:number[][] }`
   (each `number[][]` is `alphas × features`)

4. **KNN & SVR** — `knnsvr.json` *(California Housing, 1 feature vs target)*
   `k` slider → KNN step fit; SVR `epsilon`/`C` → tube + fit line.
   Schema: `{ scatter:[[x,y]], knn:[{ k, fit:[[x,y]] }], svr:[{ eps, C, fit:[[x,y]], tube:[[x,lo,hi]] }] }`

5. **Bayesian Regression** — `bayesian.json` *(Diabetes, BMI feature)*
   Prior-strength slider → mean fit + uncertainty band (±2σ).
   Schema: `{ scatter:[[x,y]], priors:[{ lambda, mean:[[x,y]], band:[[x,lo,hi]] }] }`

6. **Robust Regression** — `robust.json` *(Diabetes, BMI feature)*
   Outlier-contamination slider → OLS line swings toward injected outliers while Huber/RANSAC/
   Theil-Sen hold steady. Models: `HuberRegressor`, `RANSACRegressor`, `TheilSenRegressor`, `LinearRegression`.
   Schema: `{ scatter:[[x,y]], contamination:[{ frac, ols:[[x,y]], huber:[[x,y]], ransac:[[x,y]], theilsen:[[x,y]] }] }`

7. **Generalized Linear Models** — `glm.json` *(California Housing, positive skewed target)*
   Toggle OLS / Poisson / Gamma / Tweedie → predicted-vs-actual scatter + share of predictions that
   go negative (OLS does, the GLMs don't). Models: `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`.
   Schema: `{ actual:number[], models:[{ name, pred:number[], r2, fracNegative }], tweediePowers:[{ p, pred:number[] }] }`

8. **Quantile Regression** — `quantile.json` *(Diabetes, BMI feature)*
   Quantile slider 0.1→0.9 → fitted quantile lines stacking into a prediction interval band.
   Model: `QuantileRegressor`. Schema: `{ scatter:[[x,y]], quantiles:[{ q, fit:[[x,y]] }] }`

9. **Discriminant Analysis** — `discriminant.json` *(Wine, 2 features)*
   Toggle LDA vs QDA → decision-boundary heat grid + per-class covariance ellipses.
   Schema: `{ points:[[x,y,class]], grid:{x0,x1,y0,y1,step}, lda:number[][], qda:number[][], ellipses:{ lda:[...], qda:[...] } }`

10. **Naive Bayes** — `naivebayes.json` *(Iris, 2 features)*
    Gaussian NB class-conditional density contours + boundary.
    Schema: `{ points:[[x,y,class]], grid:{...}, boundary:number[][], densities:[...] }`

11. **KNN & SVM Boundaries** — `knnsvm.json` *(Iris, 2 features)*
    `k`, kernel (linear/rbf/poly), `C`, `gamma` → boundary heat grid reshapes.
    Schema: `{ points:[[x,y,class]], grid:{...}, knn:[{ k, grid:number[][] }], svm:[{ kernel, C, gamma, grid:number[][] }] }`

12. **Linear Classifiers** — `linclf.json` *(Iris, 2 features)*
    Perceptron `max_iter` slider → boundary settles as online learning converges; Ridge Classifier
    shown as a static least-squares reference boundary. Models: `Perceptron`, `RidgeClassifier`.
    Schema: `{ points:[[x,y,class]], grid:{...}, perceptronIters:[{ iters, grid:number[][] }], ridge:number[][] }`

13. **Boosting & Importance** — `boosting.json` *(Breast Cancer)*
    GBM + XGBoost feature importances (bar) + partial-dependence curve for the top feature.
    Schema: `{ features:string[], gbmImportance:number[], xgbImportance:number[], pdp:{ feature, x:number[], y:number[] } }`

14. **Threshold, ROC & Metrics** — `threshold.json` *(Breast Cancer)*
    Drag threshold → confusion matrix, precision/recall/F1, moving ROC + PR operating points.
    Schema: `{ scores:number[], labels:number[], roc:[[fpr,tpr]], pr:[[recall,prec]], thresholds:[{ t, tp, fp, tn, fn, precision, recall, f1 }] }`

15. **Model Comparison** — `comparison.json` *(Diabetes/Housing + Breast Cancer/Wine)*
    Capstone leaderboard: every model scored on a shared metric, sortable. Regression table
    (R²/RMSE/MAE) and classification table (accuracy/F1/ROC-AUC), each with a bar chart.
    Schema: `{ regression:[{ model, dataset, r2, rmse, mae }], classification:[{ model, dataset, accuracy, f1, rocauc }] }`

16. **About** — no artifact. Plain-language guide to every model and how the demos connect;
    links back to the two notebooks.

## Frontend behavior

- `models.ts` holds `MODEL_TABS` (id, title, tagline, dataset, notebook ref, help text) — the tab
  bar, panel header, and lazy-loaded component map all derive from it (same pattern as sample).
- Tabs lazy-load client-side (`next/dynamic`, `ssr:false`) so first paint is fast.
- Each tab: fetch its JSON once, show "loading…" until ready, then render controls + SVG.
- Light/dark aware via `globals.css`. Responsive: charts use `viewBox` + `max-width:100%`.
- Accessibility basics: tabs are real `role="tab"` buttons, sliders are `<input type="range">`
  with labels, charts have `<title>`/`aria-label`.

## XGBoost decision

Keep XGBoost in tab #9 (it's on the resume). It's only needed by the export script, which runs
once locally — not a frontend dependency. `requirements.txt` pins `xgboost`.

## Repo-polish tasks (bundled)

- `requirements.txt` — numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, jupyter (pinned to README's floors).
- `LICENSE` — MIT, author Shivani Bokka.
- `.gitignore` — Python (`__pycache__`, `.ruff_cache`, `.ipynb_checkpoints`) + Next.js (`node_modules`, `.next`, `.vercel`, `.env*`).
- Stop tracking the existing `.ruff_cache/`.
- **README** — refreshed **last** (after deploy) via the readme-writer skill, adding the live demo
  link, screenshots, and the new structure.

## Deployment

- Vercel, `web/` as the project root (Next.js autodetected).
- Build = `next build` (fully static export-friendly; no server functions).
- Artifacts are committed to `web/public/`, so the Vercel build needs no Python — it just builds
  the Next app. `export_web_artifacts.py` is run locally when data/models change.
- Deliverable: a live `*.vercel.app` URL.

## Testing / verification

- `export_web_artifacts.py` ends with an `assert`-based self-check: every expected JSON file exists,
  is valid JSON, and has non-empty top-level arrays (catches an empty/failed export before deploy).
- `web` builds clean (`next build` with no type errors).
- Manual: each tab loads its JSON and responds to its control (verified in the browser before
  calling the deploy done).

## Notebook content updates

The repo is educational-first, so the three new regression concepts are taught in the notebook,
not only computed for the frontend. `linear_regression_models.ipynb` gains three sections —
**Robust Regression** (Huber/RANSAC/Theil-Sen), **Generalized Linear Models** (Poisson/Gamma/
Tweedie), and **Quantile Regression** — each following the existing markdown-then-code-then-plot
pattern. `classification_models.ipynb` is unchanged (Logistic lives in the sibling basics repo).
The Model Comparison capstone is computed in `export_web_artifacts.py` and does not need a notebook
section.

## Build order

1. Repo polish (requirements, LICENSE, .gitignore, untrack ruff cache).
2. Add the three new sections to `linear_regression_models.ipynb`.
3. `export_web_artifacts.py` + generate all `web/public/*.json`.
4. Scaffold `web/` (package.json, config, globals.css, page.tsx, models.ts, lib helpers).
5. Build the 16 tab components (can parallelize — each is independent).
6. `next build` clean + local browser smoke test.
7. Deploy to Vercel.
8. README refresh via readme-writer skill.
