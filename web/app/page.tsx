"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { MODEL_TABS } from "./models";
import { Tip } from "./lib/tip";

const load = (p: () => Promise<{ default: React.ComponentType }>) =>
  dynamic(p, { ssr: false, loading: () => <p className="note">loading demo…</p> });

const TAB_COMPONENTS: Record<string, React.ComponentType> = {
  linear: load(() => import("./components/LinearTab")),
  biasvariance: load(() => import("./components/BiasVarianceTab")),
  regularization: load(() => import("./components/RegularizationTab")),
  knnsvr: load(() => import("./components/KnnSvrTab")),
  bayesian: load(() => import("./components/BayesianTab")),
  robust: load(() => import("./components/RobustTab")),
  glm: load(() => import("./components/GlmTab")),
  quantile: load(() => import("./components/QuantileTab")),
  discriminant: load(() => import("./components/DiscriminantTab")),
  naivebayes: load(() => import("./components/NaiveBayesTab")),
  knnsvm: load(() => import("./components/KnnSvmTab")),
  linclf: load(() => import("./components/LinClfTab")),
  boosting: load(() => import("./components/BoostingTab")),
  threshold: load(() => import("./components/ThresholdTab")),
  comparison: load(() => import("./components/ComparisonTab")),
  serving: load(() => import("./components/ServingTab")),
  about: load(() => import("./components/AboutTab")),
};

// Map each tab back to the notebook that teaches it.
const REPO = "https://github.com/shiva-shivanibokka/Dive-deeper-into-linear-models/blob/main";
const notebookFor = (group: string): string | null =>
  group === "Regression" ? `${REPO}/linear_regression_models.ipynb`
  : group === "Classification" ? `${REPO}/classification_models.ipynb`
  : null;

export default function Home() {
  const [active, setActive] = useState(MODEL_TABS[0].id);
  const tab = MODEL_TABS.find((t) => t.id === active)!;
  const Comp = TAB_COMPONENTS[tab.id];

  return (
    <main className="wrap">
      <header className="hero">
        <h1>Linear Models Playground</h1>
        <p>
          An interactive tour of the regression and classification models from the companion notebooks —
          regularization, robustness, generalized linear models, decision boundaries, and thresholds.
          Every demo runs <strong>entirely in your browser</strong> from precomputed{" "}
          <a href="https://scikit-learn.org/stable/datasets/real_world.html" target="_blank" rel="noreferrer">scikit-learn</a>{" "}
          artifacts — nothing is sent to a server.
        </p>
        <span className="live">
          <b>●</b> live · no server · precomputed on real datasets
        </span>
      </header>

      <nav className="tabs" role="tablist" aria-label="Model demos">
        {MODEL_TABS.map((t) => (
          <button key={t.id} className="tab" role="tab" aria-selected={t.id === active} onClick={() => setActive(t.id)}>
            {t.title}
          </button>
        ))}
      </nav>

      <section className="panel" role="tabpanel">
        <div className="panel-head">
          <div className="htitle">
            <h2>{tab.title}</h2>
            <Tip text={tab.help} />
          </div>
          <span className="chip">{tab.group} · {tab.dataset}</span>
        </div>
        <p className="panel-tagline">{tab.tagline}</p>
        {notebookFor(tab.group) && (
          <p className="note" style={{ marginTop: "-0.75rem", marginBottom: "1.25rem" }}>
            📓 <a href={notebookFor(tab.group)!} target="_blank" rel="noreferrer">Open the notebook that teaches this →</a>
          </p>
        )}
        {Comp ? <Comp /> : null}
      </section>

      <p className="footer">
        Built by Shivani Bokka · scikit-learn / xgboost · precomputed artifacts served client-side on Vercel
      </p>
    </main>
  );
}
