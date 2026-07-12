# Linear Models Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a deployed, interactive Next.js "Linear Models Playground" on Vercel that lets visitors explore every model in the two notebooks by moving sliders and watching inline-SVG charts redraw from precomputed JSON.

**Architecture:** Python `export_web_artifacts.py` fits sklearn models over swept parameter ranges and writes precomputed grids to `web/public/*.json`. A Next.js 15 / React 19 app reads one JSON per tab and redraws inline SVG on control changes. No ML runs in the browser; no chart libraries; no backend.

**Tech Stack:** Python 3.12 + scikit-learn + xgboost (export only); Next.js 15, React 19, TypeScript; inline SVG; Vercel hosting.

## Global Constraints

- Real `sklearn.datasets` only — no synthetic data. Datasets: California Housing, Diabetes, Iris, Wine, Breast Cancer.
- Frontend deps limited to `next`, `react`, `react-dom` (+ TS types). No chart/UI libraries.
- All demos read precomputed JSON from `web/public/`; the browser never trains or predicts.
- Artifacts are committed so the Vercel build needs zero Python.
- `random_state=42` everywhere models are fit, for reproducibility.
- Light/dark theme aware; charts use `viewBox` + `max-width:100%`; tabs/sliders are accessible (`role`, labels, `<title>`).
- Author name in copy/footer/license: **Shivani Bokka**.
- README is refreshed LAST via the readme-writer skill, after deploy.

---

## File Structure

**Repo root**
- `requirements.txt` — pinned Python deps (notebooks + export).
- `LICENSE` — MIT.
- `.gitignore` — Python + Next.js.
- `linear_regression_models.ipynb` — +3 teaching sections (Robust, GLM, Quantile).
- `scripts/export_web_artifacts.py` — the only new Python; writes all 15 JSON artifacts.

**web/** (Next.js)
- `package.json`, `next.config.mjs`, `tsconfig.json`, `next-env.d.ts`
- `app/globals.css` — theme + layout.
- `app/models.ts` — `MODEL_TABS` registry (single source of truth).
- `app/lib/svg.tsx` — SVG chart primitives shared by all tabs.
- `app/lib/tip.tsx` — "?" tooltip.
- `app/lib/data.ts` — `useArtifact(name)` fetch hook.
- `app/page.tsx` — hero + tab bar + panel shell.
- `app/components/*Tab.tsx` — 16 tab components (15 data + About).
- `public/*.json` — 15 artifacts (About has none).

---

## Task 1: Repo polish

**Files:**
- Create: `requirements.txt`, `LICENSE`, `.gitignore`

- [ ] **Step 1: Write `requirements.txt`**

```
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
xgboost>=2.0
jupyter>=1.0
```

- [ ] **Step 2: Write `LICENSE`** — standard MIT text, `Copyright (c) 2026 Shivani Bokka`.

- [ ] **Step 3: Write `.gitignore`**

```
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
.ruff_cache/
.venv/
# Next.js
web/node_modules/
web/.next/
web/out/
web/.vercel/
.env*
```

- [ ] **Step 4: Stop tracking ruff cache**

Run: `git rm -r --cached .ruff_cache 2>/dev/null; git status`
Expected: `.ruff_cache` no longer staged/tracked.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt LICENSE .gitignore
git commit -m "chore: add requirements, license, gitignore"
```

---

## Task 2: Notebook teaching sections

**Files:**
- Modify: `linear_regression_models.ipynb` (append 3 markdown+code+plot sections)

**Interfaces:**
- Produces: nothing consumed by code; these are teaching cells. Uses the same imports already
  present in the notebook (numpy, matplotlib, sklearn, `fetch_california_housing`, `load_diabetes`).

- [ ] **Step 1: Add "Robust Regression" section** — markdown explainer (why OLS is sensitive to
  outliers; Huber/RANSAC/Theil-Sen) + code fitting `LinearRegression`, `HuberRegressor`,
  `RANSACRegressor`, `TheilSenRegressor` on Diabetes BMI feature with a few injected outliers,
  plus a scatter+lines plot.

- [ ] **Step 2: Add "Generalized Linear Models" section** — markdown (link function, non-Gaussian
  targets) + code fitting `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor` vs OLS on
  California Housing, printing R² and share-of-negative-predictions, plus predicted-vs-actual plot.

- [ ] **Step 3: Add "Quantile Regression" section** — markdown (pinball loss, prediction intervals)
  + code fitting `QuantileRegressor` at q=0.1/0.5/0.9 on Diabetes BMI, plus a band plot.

- [ ] **Step 4: Run the notebook top-to-bottom** to confirm no errors and outputs render.

Run: `cd "<repo>" && jupyter nbconvert --to notebook --execute --inplace linear_regression_models.ipynb`
Expected: exit 0, new cells have outputs.

- [ ] **Step 5: Commit**

```bash
git add linear_regression_models.ipynb
git commit -m "docs: add robust, GLM, and quantile regression sections"
```

---

## Task 3: Export script + all artifacts

**Files:**
- Create: `scripts/export_web_artifacts.py`
- Create (generated): `web/public/*.json` (15 files)

**Interfaces:**
- Produces: the 15 JSON files whose schemas are the contracts consumed by Tasks 6–20. Schemas are
  copied verbatim from the design spec's "Tabs & artifact contracts" section.

**Design of the script:** one `def build_<name>() -> dict` per artifact, a `WRITERS` dict mapping
filename → builder, a loop that writes each `web/public/<name>.json`, and a final self-check.

- [ ] **Step 1: Write the script skeleton + self-check**

```python
"""Precompute model artifacts for the web playground. Run locally when data/models change."""
import json, os
from pathlib import Path
import numpy as np
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                              load_iris, load_wine, load_breast_cancer)

OUT = Path(__file__).resolve().parent.parent / "web" / "public"
RNG = 42

def _line(model, xs):
    return [[float(x), float(model.predict([[x]])[0])] for x in xs]

# ... one build_<name>() per artifact (Steps 2-16) ...

WRITERS = {}  # filled as builders are defined; name -> callable

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in WRITERS.items():
        data = fn()
        (OUT / f"{name}.json").write_text(json.dumps(data))
        print("wrote", name, len(json.dumps(data)), "bytes")
    # self-check: every file exists, valid JSON, non-empty
    for name in WRITERS:
        obj = json.loads((OUT / f"{name}.json").read_text())
        assert obj, f"{name}.json is empty"
    print("OK", len(WRITERS), "artifacts")

if __name__ == "__main__":
    main()
```

- [ ] **Steps 2–16: Implement one builder per artifact**, each registered in `WRITERS`. Each builder
  fits the models named in the spec with `random_state=42`, sweeps the control parameter, and returns
  a dict exactly matching the spec schema. One builder per file:
  `linear, biasvariance, regularization, knnsvr, bayesian, robust, glm, quantile, discriminant,
  naivebayes, knnsvm, linclf, boosting, threshold, comparison`.
  Boundary/heat grids: evaluate the fitted model on a `grid.step`-spaced mesh over the 2-feature
  range and return `number[][]` of predicted class (ints). Keep grids ~60×60 to bound file size.

- [ ] **Step 17: Run the script**

Run: `cd "<repo>" && python scripts/export_web_artifacts.py`
Expected: prints "wrote ..." 15 times then `OK 15 artifacts`, exit 0.

- [ ] **Step 18: Sanity-check a couple files**

Run: `python -c "import json;d=json.load(open('web/public/regularization.json'));print(len(d['alphas']),len(d['lasso']))"`
Expected: matching non-zero lengths (alphas count == rows of lasso path).

- [ ] **Step 19: Commit**

```bash
git add scripts/export_web_artifacts.py web/public/*.json
git commit -m "feat: export precomputed model artifacts for web"
```

---

## Task 4: Web scaffold (config + theme + registry + shared libs)

**Files:**
- Create: `web/package.json`, `web/next.config.mjs`, `web/tsconfig.json`, `web/next-env.d.ts`,
  `web/app/globals.css`, `web/app/models.ts`, `web/app/lib/svg.tsx`, `web/app/lib/tip.tsx`,
  `web/app/lib/data.ts`

**Interfaces:**
- Produces:
  - `MODEL_TABS: ModelTab[]` with `{id,nb,title,tagline,dataset,help}` (mirrors sample's shape).
  - `useArtifact<T>(name: string): T | null` — client hook: fetches `/${name}.json` once, returns
    null while loading.
  - SVG helpers in `svg.tsx`:
    - `Chart({width?,height?,children,title})` — responsive `<svg viewBox>` wrapper with `<title>`.
    - `useScale(domain:[number,number], range:[number,number]) => (v:number)=>number`
    - `Axes({sx,sy,x0,x1,y0,y1})`, `Line({pts,sx,sy,stroke?})`, `Scatter({pts,sx,sy,color?})`,
      `Band({pts,sx,sy})` (pts=`[x,lo,hi][]`), `Heat({grid,x0,x1,y0,y1,colors})`, `Bars({values,labels})`.
  - `Tip({text})` — the sample's tooltip component.

- [ ] **Step 1: `package.json`** — copy the sample's exactly (name → `linear-models-playground`,
  deps `next ^15.5.20`, `react 19.0.0`, `react-dom 19.0.0`; devDeps TS + types as in sample).

- [ ] **Step 2: `next.config.mjs`, `tsconfig.json`, `next-env.d.ts`** — copy from the sample repo.

- [ ] **Step 3: `app/lib/data.ts`**

```tsx
"use client";
import { useEffect, useState } from "react";
export function useArtifact<T>(name: string): T | null {
  const [d, setD] = useState<T | null>(null);
  useEffect(() => { fetch(`/${name}.json`).then(r => r.json()).then(setD).catch(() => setD(null)); }, [name]);
  return d;
}
```

- [ ] **Step 4: `app/lib/svg.tsx`** — implement the helper API above. `Heat` renders one `<rect>` per
  grid cell colored by class; `Line`/`Scatter`/`Band`/`Bars`/`Axes` are thin SVG. `useScale` returns a
  linear mapper. Keep it dependency-free.

- [ ] **Step 5: `app/lib/tip.tsx`** — copy from sample.

- [ ] **Step 6: `app/models.ts`** — `MODEL_TABS` with all 16 entries (ids: `linear, biasvariance,
  regularization, knnsvr, bayesian, robust, glm, quantile, discriminant, naivebayes, knnsvm, linclf,
  boosting, threshold, comparison, about`), each with title/tagline/dataset/help from the spec.

- [ ] **Step 7: `app/globals.css`** — adapt the sample's CSS (hero, tabs, panel, chip, controls,
  footer) with light/dark via `prefers-color-scheme`. Add `.control{}` row styling for sliders/toggles.

- [ ] **Step 8: Install + typecheck**

Run: `cd web && npm install && npx tsc --noEmit`
Expected: install succeeds; tsc reports no errors (tabs not imported yet).

- [ ] **Step 9: Commit**

```bash
git add web/package.json web/next.config.mjs web/tsconfig.json web/next-env.d.ts web/app/globals.css web/app/models.ts web/app/lib
git commit -m "feat: scaffold web app shell, registry, and svg helpers"
```

---

## Task 5: Page shell + About tab (app builds and runs)

**Files:**
- Create: `web/app/page.tsx`, `web/app/components/AboutTab.tsx`

**Interfaces:**
- Consumes: `MODEL_TABS`, `Tip`. Lazy-loads each `components/<Id>Tab` via `next/dynamic({ssr:false})`.
- Produces: the running tab shell. Tab components are registered in a `TAB_COMPONENTS` map keyed by id.

- [ ] **Step 1: `page.tsx`** — adapt the sample's `page.tsx`: hero title "Linear Models Playground",
  tagline crediting the notebooks + client-side-on-Vercel, tab bar from `MODEL_TABS`, panel with the
  active tab's title/chip/tagline/help, and the lazy `TAB_COMPONENTS` map with an entry per id.

- [ ] **Step 2: `AboutTab.tsx`** — static JSX: plain-language one-liners for every model grouped by
  Regression / Classification / Capstone, and a note that each demo reads precomputed artifacts.

- [ ] **Step 3: Temporary stubs** — for the 14 not-yet-built data tabs, create one-line placeholder
  components (`export default () => <p className="note">coming soon</p>`) so the map resolves and the
  app builds. (Each real tab task replaces its stub.)

- [ ] **Step 4: Dev-run smoke test**

Run: `cd web && npm run build`
Expected: build succeeds; About tab renders.

- [ ] **Step 5: Commit**

```bash
git add web/app/page.tsx web/app/components
git commit -m "feat: page shell + About tab + tab stubs"
```

---

## Reference tab pattern (Tasks 6–20 all follow this)

Every data tab is the same shape. Task 6 below is fully coded as the reference; Tasks 7–20 each state
only their **artifact**, **controls**, and **what they draw**, and reuse this exact structure.

```tsx
"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, useScale, Axes, Line, Scatter /* ...as needed */ } from "../lib/svg";

type Art = { /* the tab's schema from the spec */ };

export default function XxxTab() {
  const d = useArtifact<Art>("xxx");
  const [i, setI] = useState(0);            // index into the swept-parameter array
  if (!d) return <p className="note">loading…</p>;
  const frame = d.someSweptArray[i];
  const sx = useScale([xmin, xmax], [40, 360]);
  const sy = useScale([ymin, ymax], [220, 20]);
  return (
    <>
      <div className="control">
        <label htmlFor="p">Param: {frame.label}</label>
        <input id="p" type="range" min={0} max={d.someSweptArray.length - 1}
               value={i} onChange={e => setI(+e.target.value)} />
      </div>
      <Chart title="Xxx demo">
        <Axes sx={sx} sy={sy} x0={xmin} x1={xmax} y0={ymin} y1={ymax} />
        <Scatter pts={d.scatter} sx={sx} sy={sy} />
        <Line pts={frame.fit} sx={sx} sy={sy} />
      </Chart>
    </>
  );
}
```

Each tab task's steps are identical in form:
1. Replace the stub `web/app/components/<Id>Tab.tsx` with the real component consuming `<id>.json`.
2. `cd web && npx tsc --noEmit` → no errors.
3. `npm run build` → succeeds.
4. Commit `feat: <id> tab`.

---

## Task 6: Linear tab (reference — fully coded)

**Files:** Modify `web/app/components/LinearTab.tsx` · Artifact `linear.json`

- [ ] **Step 1: Implement**

```tsx
"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, useScale, Axes, Line, Scatter } from "../lib/svg";

type PF = { coef: number; r2: number; rmse: number; points: [number, number][]; fit: [number, number][] };
type Art = { features: string[]; perFeature: Record<string, PF> };

export default function LinearTab() {
  const d = useArtifact<Art>("linear");
  const [f, setF] = useState<string | null>(null);
  if (!d) return <p className="note">loading…</p>;
  const feat = f ?? d.features[0];
  const pf = d.perFeature[feat];
  const xs = pf.points.map(p => p[0]), ys = pf.points.map(p => p[1]);
  const sx = useScale([Math.min(...xs), Math.max(...xs)], [44, 360]);
  const sy = useScale([Math.min(...ys), Math.max(...ys)], [220, 20]);
  return (
    <>
      <div className="control">
        <label htmlFor="feat">Feature:</label>
        <select id="feat" value={feat} onChange={e => setF(e.target.value)}>
          {d.features.map(n => <option key={n}>{n}</option>)}
        </select>
        <span className="chip">R² {pf.r2.toFixed(3)} · RMSE {pf.rmse.toFixed(2)}</span>
      </div>
      <Chart title={`Linear fit on ${feat}`}>
        <Axes sx={sx} sy={sy} x0={Math.min(...xs)} x1={Math.max(...xs)} y0={Math.min(...ys)} y1={Math.max(...ys)} />
        <Scatter pts={pf.points} sx={sx} sy={sy} />
        <Line pts={pf.fit} sx={sx} sy={sy} stroke="var(--accent)" />
      </Chart>
    </>
  );
}
```

- [ ] **Step 2:** `cd web && npx tsc --noEmit` → no errors.
- [ ] **Step 3:** `npm run build` → succeeds.
- [ ] **Step 4:** Commit `feat: linear regression tab`.

---

## Tasks 7–20: remaining data tabs

Each follows the reference pattern. Per-tab specifics (artifact / controls / marks):

- **Task 7 — BiasVarianceTab** (`biasvariance.json`): control = degree slider (index into `degrees`);
  draw `scatter` + `degrees[i].fit` line; side readout of `trainErr`/`testErr`; a second small
  Line chart of train vs test error across all degrees with a marker at `i`.
- **Task 8 — RegularizationTab** (`regularization.json`): control = alpha slider + model toggle
  (ridge/lasso/elastic); draw one `Line` per feature across `alphas` with a vertical marker at the
  selected alpha (coefficient paths); highlight coefficients driven to 0 for lasso.
- **Task 9 — KnnSvrTab** (`knnsvr.json`): controls = `k` slider (KNN) and `eps`/`C` selectors (SVR)
  with a KNN/SVR toggle; draw `scatter` + selected `fit`; for SVR also draw `Band` from `tube`.
- **Task 10 — BayesianTab** (`bayesian.json`): control = prior-strength slider (index into `priors`);
  draw `scatter` + `priors[i].mean` line + `Band` from `priors[i].band`.
- **Task 11 — RobustTab** (`robust.json`): control = contamination slider (index into `contamination`);
  draw `scatter` + four lines (ols/huber/ransac/theilsen) with a legend; OLS in a warning color to
  show it swinging.
- **Task 12 — GlmTab** (`glm.json`): control = model toggle (ols/poisson/gamma/tweedie); draw
  predicted-vs-actual `Scatter` (`actual` vs `models[k].pred`) + diagonal reference `Line`; chip shows
  `r2` and `fracNegative`.
- **Task 13 — QuantileTab** (`quantile.json`): control = quantile slider (index into `quantiles`);
  draw `scatter` + selected quantile `fit`; optionally overlay q=0.1 and q=0.9 as a faint band.
- **Task 14 — DiscriminantTab** (`discriminant.json`): control = LDA/QDA toggle; draw `Heat`
  from `lda`/`qda` grid + `Scatter` of `points` colored by class + covariance `ellipses`.
- **Task 15 — NaiveBayesTab** (`naivebayes.json`): draw `Heat` from `boundary` + `Scatter` points;
  no slider (static concept view) or a class-density opacity toggle.
- **Task 16 — KnnSvmTab** (`knnsvm.json`): controls = `k` slider (KNN) and kernel/`C`/`gamma`
  selectors (SVM) with a KNN/SVM toggle; draw the selected `grid` as `Heat` + `Scatter` points.
- **Task 17 — LinClfTab** (`linclf.json`): control = Perceptron `iters` slider (index into
  `perceptronIters`) + a "show Ridge Classifier" toggle; draw selected `Heat` + `Scatter` points.
- **Task 18 — BoostingTab** (`boosting.json`): control = GBM/XGBoost toggle; draw `Bars` of the
  selected importances (top ~12 features) + a `Line` of the `pdp` curve.
- **Task 19 — ThresholdTab** (`threshold.json`): control = threshold slider (index into `thresholds`);
  draw a confusion-matrix 2×2 grid (tp/fp/tn/fn) + precision/recall/F1 chips + ROC `Line` with a
  moving operating-point dot + PR `Line` with its dot.
- **Task 20 — ComparisonTab** (`comparison.json`): control = metric selector + table sort; render two
  sortable tables (regression: r2/rmse/mae; classification: accuracy/f1/rocauc) + a `Bars` chart of the
  selected metric.

Each: implement → `npx tsc --noEmit` → `npm run build` → commit `feat: <id> tab`.

---

## Task 21: Full build + local smoke test

- [ ] **Step 1:** `cd web && npm run build` → succeeds, all 16 tabs compiled, no stubs left.
- [ ] **Step 2:** `npm run start` and click through every tab; each loads its JSON and its control
  moves the chart. Note any tab that fails and fix before proceeding.
- [ ] **Step 3:** Commit any fixes `fix: <tab> smoke-test issues`.

---

## Task 22: Deploy to Vercel

- [ ] **Step 1:** Confirm deploy target with the user (Vercel, `web/` as root).
- [ ] **Step 2:** Deploy `web/` to Vercel (production). Root directory = `web`, framework = Next.js,
  no env vars, no Python build step.
- [ ] **Step 3:** Open the live URL and click through 2–3 tabs to confirm the deploy works.
- [ ] **Step 4:** Record the live URL for the README.

---

## Task 23: README refresh (LAST)

- [ ] **Step 1:** Invoke the readme-writer skill to regenerate `README.md`, adding: live demo link,
  the 16-demo overview, screenshots, the new repo structure (`scripts/`, `web/`), and the new notebook
  sections. Keep the existing concept explanations that are still accurate.
- [ ] **Step 2:** Commit `docs: refresh README with live demo and playground`.

---

## Self-Review

- **Spec coverage:** every spec tab (16) maps to a task (5 for About, 6 for Linear, 7–20 for the rest);
  export script → Task 3; notebook sections → Task 2; repo polish → Task 1; deploy → Task 22; README →
  Task 23. Datasets, no-synthetic, dep limits, committed-artifacts, reproducibility → Global Constraints.
- **Placeholders:** the 15 export builders and 14 tab bodies are specified by exact schema + controls +
  marks rather than inlined verbatim — deliberate DRY against the reference pattern, not a TODO. All
  shared infra (data hook, svg API, page shell, reference tab) is fully coded.
- **Type consistency:** artifact schemas are the single contract, copied from the spec; each tab's
  `type Art` mirrors its `<id>.json`. `useArtifact`, `MODEL_TABS`, and the svg helper names are used
  consistently across tasks.
