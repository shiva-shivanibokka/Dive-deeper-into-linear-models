"use client";
import { MODEL_TABS } from "../models";

const BLURB: Record<string, string> = {
  linear: "Ordinary least squares — the straight line minimizing squared error. The baseline every other model is measured against.",
  biasvariance: "Polynomial regression dialed from too-simple to too-complex, making the bias–variance trade-off visible as train and test error split apart.",
  regularization: "Ridge, Lasso, and ElasticNet — penalties that shrink coefficients to fight overfitting, with Lasso doubling as automatic feature selection.",
  knnsvr: "K-nearest-neighbors regression (local averaging) and support-vector regression (an ε-insensitive tube), two non-parametric ways to fit a curve.",
  bayesian: "Bayesian linear regression returns a distribution over fits, giving calibrated uncertainty bands that widen where data is sparse.",
  robust: "Huber, RANSAC, and Theil-Sen — regressions built to ignore outliers that would drag ordinary least squares off course.",
  glm: "Poisson, Gamma, and Tweedie regression — generalized linear models that respect positive and skewed targets instead of assuming Gaussian noise.",
  quantile: "Quantile regression predicts percentiles rather than the mean, turning a point estimate into a full prediction interval.",
  discriminant: "LDA and QDA — Gaussian classifiers whose shared-vs-per-class covariance assumption gives straight or curved boundaries.",
  naivebayes: "Gaussian Naive Bayes — Bayes' theorem with a strong feature-independence assumption that is fast and surprisingly competitive.",
  knnsvm: "K-nearest-neighbors and support-vector classification, drawing boundaries by local vote or by maximum-margin separation with a kernel.",
  linclf: "The Perceptron (online, mistake-driven) and the Ridge Classifier (one-shot least squares) — two more ways to draw a linear boundary.",
  boosting: "Gradient Boosting and XGBoost — additive ensembles of small trees, shown here through feature importance and partial dependence.",
  threshold: "Turning probabilities into decisions: the threshold trades precision against recall, traced live on the ROC and PR curves.",
  comparison: "Every model benchmarked on the same held-out split, so you can see which family wins on which task.",
};

export default function AboutTab() {
  const groups = ["Regression", "Classification", "Capstone"] as const;
  return (
    <div className="about">
      <p className="callout">
        Every demo here reads a small <strong>precomputed JSON artifact</strong> exported from the notebooks by{" "}
        <code>scripts/export_web_artifacts.py</code>. The models are trained once in Python on real{" "}
        <span className="k">scikit-learn</span> datasets; the browser only redraws the results as you move the
        controls. No model runs client-side, so the whole playground is static and instant.
      </p>
      {groups.map((g) => (
        <div key={g}>
          <h3><span className="num">§</span>{g}</h3>
          {MODEL_TABS.filter((t) => t.group === g && t.id !== "about").map((t) => (
            <p key={t.id}><span className="k">{t.title}</span> — {BLURB[t.id]}</p>
          ))}
        </div>
      ))}
      <p className="note" style={{ marginTop: "1.5rem" }}>
        Linear and logistic regression <em>fundamentals</em> (pipelines, leakage-free preprocessing, evaluation) live
        in the companion repo <strong>Basics of Linear &amp; Logistic Regression</strong>. This playground goes deeper
        into the regularized, kernel, robust, generalized, and probabilistic members of the family.
      </p>
    </div>
  );
}
