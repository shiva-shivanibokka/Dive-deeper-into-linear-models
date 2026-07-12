"""Precompute model artifacts for the Linear Models Playground web app.

Fits scikit-learn / xgboost models over swept parameter ranges on real sklearn
datasets and writes precomputed JSON grids to ``web/public/*.json``. The browser
never trains or predicts — it only redraws these artifacts. Run this locally
whenever the data or models change; the committed JSON is what Vercel serves.

Usage:  python scripts/export_web_artifacts.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                              load_iris, load_wine, load_breast_cancer)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  BayesianRidge, HuberRegressor, RANSACRegressor,
                                  TheilSenRegressor, PoissonRegressor,
                                  GammaRegressor, TweedieRegressor,
                                  QuantileRegressor, Perceptron, RidgeClassifier)
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR, SVC
from xgboost import XGBClassifier

RNG = 42
OUT = Path(__file__).resolve().parent.parent / "web" / "public"


# ---------------------------------------------------------------- helpers
def r(v, n=5):
    return round(float(v), n)


def pts(x, y, n=5):
    return [[r(a, n), r(b, n)] for a, b in zip(x, y)]


def sub(*arrays, n=300, seed=0):
    """Subsample paired arrays down to n points to bound JSON size."""
    m = len(arrays[0])
    if m <= n:
        return arrays
    idx = np.random.default_rng(seed).choice(m, n, replace=False)
    return tuple(a[idx] for a in arrays)


def line_over(model, xlo, xhi, k=80):
    xs = np.linspace(xlo, xhi, k)
    ys = model.predict(xs.reshape(-1, 1))
    return pts(xs, ys)


def class_grid(model, x0, x1, y0, y1, step=50):
    xs = np.linspace(x0, x1, step)
    ys = np.linspace(y0, y1, step)
    xx, yy = np.meshgrid(xs, ys)
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
    return z.reshape(step, step).tolist()


def ellipse(mean, cov, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    rx, ry = (n_std * np.sqrt(np.maximum(vals, 1e-9))).tolist()
    return {"cx": r(mean[0]), "cy": r(mean[1]), "rx": r(rx), "ry": r(ry), "angle": r(angle)}


def two_feature_setup(X, y, i, j):
    """Standardize two chosen features; return standardized points and bounds."""
    Z = StandardScaler().fit_transform(X[:, [i, j]])
    x0, x1 = float(Z[:, 0].min()) - 0.5, float(Z[:, 0].max()) + 0.5
    y0, y1 = float(Z[:, 1].min()) - 0.5, float(Z[:, 1].max()) + 0.5
    return Z, (x0, x1, y0, y1)


# ---------------------------------------------------------------- builders
def build_linear():
    d = fetch_california_housing()
    X, y, names = d.data, d.target, list(d.feature_names)
    per = {}
    for i, name in enumerate(names):
        xi = X[:, i]
        m = LinearRegression().fit(xi.reshape(-1, 1), y)
        pred = m.predict(xi.reshape(-1, 1))
        xs, ys = sub(xi, y, n=280, seed=i)
        per[name] = {
            "coef": r(m.coef_[0]), "r2": r(r2_score(y, pred)),
            "rmse": r(np.sqrt(mean_squared_error(y, pred))),
            "points": pts(xs, ys),
            "fit": line_over(m, float(xi.min()), float(xi.max()), 2),
        }
    return {"features": names, "perFeature": per}


def build_biasvariance():
    d = load_diabetes()
    x = StandardScaler().fit_transform(d.data[:, 2:3]).ravel()  # BMI, standardized
    y = d.target
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=RNG)
    scatter = pts(x, y)
    xg = np.linspace(x.min(), x.max(), 80)
    degrees = []
    for deg in range(1, 16):
        m = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        m.fit(xtr.reshape(-1, 1), ytr)
        fit = pts(xg, m.predict(xg.reshape(-1, 1)))
        tr = np.sqrt(mean_squared_error(ytr, m.predict(xtr.reshape(-1, 1))))
        te = np.sqrt(mean_squared_error(yte, m.predict(xte.reshape(-1, 1))))
        degrees.append({"degree": deg, "fit": fit, "trainErr": r(tr, 2), "testErr": r(te, 2)})
    return {"scatter": scatter, "degrees": degrees}


def build_regularization():
    d = load_diabetes()
    X = StandardScaler().fit_transform(d.data)
    y = d.target
    names = list(d.feature_names)
    alphas = np.logspace(-2, 2.3, 30)
    ridge, lasso, elastic = [], [], []
    for a in alphas:
        ridge.append([r(c, 4) for c in Ridge(alpha=a).fit(X, y).coef_])
        lasso.append([r(c, 4) for c in Lasso(alpha=a, max_iter=5000).fit(X, y).coef_])
        elastic.append([r(c, 4) for c in ElasticNet(alpha=a, max_iter=5000).fit(X, y).coef_])
    return {"features": names, "alphas": [r(a, 4) for a in alphas],
            "ridge": ridge, "lasso": lasso, "elastic": elastic}


def build_knnsvr():
    d = fetch_california_housing()
    x = d.data[:, 0]  # MedInc
    y = d.target
    keep = x < 12  # trim a few extreme-income outliers for a cleaner view
    x, y = x[keep], y[keep]
    xs = StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
    xg = np.linspace(xs.min(), xs.max(), 80)
    sx, sy = sub(xs, y, n=300, seed=1)
    scatter = pts(sx, sy)
    knn = []
    for k in (1, 5, 15, 50):
        m = KNeighborsRegressor(k).fit(xs.reshape(-1, 1), y)
        knn.append({"k": k, "fit": pts(xg, m.predict(xg.reshape(-1, 1)))})
    svr = []
    for eps in (0.1, 0.5):
        for C in (1.0, 10.0):
            m = SVR(kernel="rbf", epsilon=eps, C=C).fit(xs.reshape(-1, 1), y)
            yg = m.predict(xg.reshape(-1, 1))
            tube = [[r(a), r(b - eps), r(b + eps)] for a, b in zip(xg, yg)]
            svr.append({"eps": eps, "C": C, "fit": pts(xg, yg), "tube": tube})
    return {"scatter": scatter, "knn": knn, "svr": svr}


def build_bayesian():
    """Closed-form Bayesian linear regression on polynomial features of BMI.

    Prior on weights N(0, (1/lambda) I); noise precision beta from data. Sweeping
    lambda shows the prior pulling the fit toward zero and widening the band away
    from data.
    """
    d = load_diabetes()
    x = StandardScaler().fit_transform(d.data[:, 2:3]).ravel()
    y = d.target.astype(float)
    deg = 5
    Phi = PolynomialFeatures(deg).fit_transform(x.reshape(-1, 1))
    beta = 1.0 / np.var(y)
    xg = np.linspace(x.min(), x.max(), 80)
    Pg = PolynomialFeatures(deg).fit_transform(xg.reshape(-1, 1))
    priors = []
    for lam in (0.5, 5.0, 50.0, 500.0):
        S = np.linalg.inv(lam * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)
        m = beta * S @ Phi.T @ y
        mean = Pg @ m
        var = 1.0 / beta + np.einsum("ij,jk,ik->i", Pg, S, Pg)
        sd = np.sqrt(np.maximum(var, 0))
        band = [[r(a), r(mu - 2 * s), r(mu + 2 * s)] for a, mu, s in zip(xg, mean, sd)]
        priors.append({"lambda": lam, "mean": pts(xg, mean), "band": band})
    return {"scatter": pts(x, y), "priors": priors}


def build_robust():
    d = load_diabetes()
    x = StandardScaler().fit_transform(d.data[:, 2:3]).ravel()
    y = d.target.astype(float)
    scatter = pts(x, y)
    xlo, xhi = float(x.min()), float(x.max())
    rng = np.random.default_rng(RNG)
    contamination = []
    for frac in (0.0, 0.05, 0.1, 0.2):
        xo, yo = x.copy(), y.copy()
        out_pts = []
        n_out = int(frac * len(x))
        if n_out:
            idx = rng.choice(len(x), n_out, replace=False)
            yo = yo.copy()
            yo[idx] = yo[idx] + rng.uniform(300, 500, n_out)  # gross y-outliers
            out_pts = pts(xo[idx], yo[idx])
        fits = {}
        for key, model in (("ols", LinearRegression()),
                           ("huber", HuberRegressor()),
                           ("ransac", RANSACRegressor(random_state=RNG)),
                           ("theilsen", TheilSenRegressor(random_state=RNG))):
            model.fit(xo.reshape(-1, 1), yo)
            fits[key] = line_over(model, xlo, xhi, 2)
        contamination.append({"frac": frac, "outliers": out_pts, **fits})
    return {"scatter": scatter, "contamination": contamination}


def build_glm():
    d = fetch_california_housing()
    X, y = d.data, d.target  # target > 0 (100k USD)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RNG)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xte_s, yte_s = sub(Xte, yte, n=300, seed=2)
    models = []
    specs = {"OLS": LinearRegression(),
             "Poisson": PoissonRegressor(max_iter=500),
             "Gamma": GammaRegressor(max_iter=500),
             "Tweedie(1.5)": TweedieRegressor(power=1.5, max_iter=500)}
    for name, m in specs.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte_s)
        models.append({"name": name, "pred": [r(p) for p in pred],
                       "r2": r(r2_score(yte_s, pred)),
                       "fracNegative": r(float(np.mean(pred < 0)))})
    tw = []
    for p in (0.0, 1.0, 1.5, 2.0, 3.0):
        m = TweedieRegressor(power=p, max_iter=500).fit(Xtr, ytr)
        tw.append({"p": p, "pred": [r(v) for v in m.predict(Xte_s)]})
    return {"actual": [r(v) for v in yte_s], "models": models, "tweediePowers": tw}


def build_quantile():
    d = load_diabetes()
    x = StandardScaler().fit_transform(d.data[:, 2:3]).ravel()
    y = d.target.astype(float)
    xg = np.linspace(x.min(), x.max(), 60)
    quantiles = []
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        m = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
        m.fit(x.reshape(-1, 1), y)
        quantiles.append({"q": q, "fit": pts(xg, m.predict(xg.reshape(-1, 1)))})
    return {"scatter": pts(x, y), "quantiles": quantiles}


def build_discriminant():
    d = load_wine()
    i, j = 6, 9  # flavanoids, color_intensity — separable pair
    Z, (x0, x1, y0, y1) = two_feature_setup(d.data, d.target, i, j)
    y = d.target
    grid = {"x0": r(x0), "x1": r(x1), "y0": r(y0), "y1": r(y1), "step": 50}
    lda = LinearDiscriminantAnalysis().fit(Z, y)
    qda = QuadraticDiscriminantAnalysis().fit(Z, y)
    classes = np.unique(y)
    pooled = np.mean([np.cov(Z[y == c].T) for c in classes], axis=0)
    ell_lda = [ellipse(Z[y == c].mean(0), pooled) for c in classes]
    ell_qda = [ellipse(Z[y == c].mean(0), np.cov(Z[y == c].T)) for c in classes]
    points = [[r(a), r(b), int(c)] for (a, b), c in zip(Z, y)]
    return {"points": points, "grid": grid,
            "lda": class_grid(lda, x0, x1, y0, y1),
            "qda": class_grid(qda, x0, x1, y0, y1),
            "ellipses": {"lda": ell_lda, "qda": ell_qda}}


def build_naivebayes():
    d = load_iris()
    i, j = 2, 3  # petal length, petal width
    Z, (x0, x1, y0, y1) = two_feature_setup(d.data, d.target, i, j)
    y = d.target
    grid = {"x0": r(x0), "x1": r(x1), "y0": r(y0), "y1": r(y1), "step": 50}
    nb = GaussianNB().fit(Z, y)
    points = [[r(a), r(b), int(c)] for (a, b), c in zip(Z, y)]
    return {"points": points, "grid": grid, "boundary": class_grid(nb, x0, x1, y0, y1)}


def build_knnsvm():
    d = load_iris()
    i, j = 2, 3
    Z, (x0, x1, y0, y1) = two_feature_setup(d.data, d.target, i, j)
    y = d.target
    grid = {"x0": r(x0), "x1": r(x1), "y0": r(y0), "y1": r(y1), "step": 50}
    points = [[r(a), r(b), int(c)] for (a, b), c in zip(Z, y)]
    knn = [{"k": k, "grid": class_grid(KNeighborsClassifier(k).fit(Z, y), x0, x1, y0, y1)}
           for k in (1, 5, 15)]
    svm = []
    for kernel, C, gamma in (("linear", 1.0, "scale"), ("rbf", 1.0, "scale"), ("poly", 1.0, "scale")):
        m = SVC(kernel=kernel, C=C, gamma=gamma).fit(Z, y)
        svm.append({"kernel": kernel, "C": C, "gamma": str(gamma),
                    "grid": class_grid(m, x0, x1, y0, y1)})
    return {"points": points, "grid": grid, "knn": knn, "svm": svm}


def build_linclf():
    d = load_iris()
    i, j = 2, 3
    Z, (x0, x1, y0, y1) = two_feature_setup(d.data, d.target, i, j)
    y = d.target
    grid = {"x0": r(x0), "x1": r(x1), "y0": r(y0), "y1": r(y1), "step": 50}
    points = [[r(a), r(b), int(c)] for (a, b), c in zip(Z, y)]
    per = []
    for it in (1, 2, 5, 20):
        m = Perceptron(max_iter=it, tol=None, random_state=RNG).fit(Z, y)
        per.append({"iters": it, "grid": class_grid(m, x0, x1, y0, y1)})
    ridge = class_grid(RidgeClassifier().fit(Z, y), x0, x1, y0, y1)
    return {"points": points, "grid": grid, "perceptronIters": per, "ridge": ridge}


def build_boosting():
    d = load_breast_cancer()
    X, y, names = d.data, d.target, list(d.feature_names)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RNG, stratify=y)
    gbm = GradientBoostingClassifier(random_state=RNG).fit(Xtr, ytr)
    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                        eval_metric="logloss", random_state=RNG).fit(Xtr, ytr)
    top = int(np.argmax(gbm.feature_importances_))
    pd = partial_dependence(gbm, Xtr, [top], grid_resolution=40)
    return {"features": names,
            "gbmImportance": [r(v, 5) for v in gbm.feature_importances_],
            "xgbImportance": [r(v, 5) for v in xgb.feature_importances_],
            "pdp": {"feature": names[top],
                    "x": [r(v) for v in pd["grid_values"][0]],
                    "y": [r(v) for v in pd["average"][0]]}}


def build_threshold():
    d = load_breast_cancer()
    X, y = d.data, d.target  # 1 = benign (positive)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
    m = GradientBoostingClassifier(random_state=RNG).fit(Xtr, ytr)
    scores = m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, scores)
    prec, rec, _ = precision_recall_curve(yte, scores)
    thr = []
    for t in np.linspace(0.02, 0.98, 49):
        pred = (scores >= t).astype(int)
        tp = int(np.sum((pred == 1) & (yte == 1)))
        fp = int(np.sum((pred == 1) & (yte == 0)))
        tn = int(np.sum((pred == 0) & (yte == 0)))
        fn = int(np.sum((pred == 0) & (yte == 1)))
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        thr.append({"t": r(t, 3), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "precision": r(p), "recall": r(rc), "f1": r(f1)})
    return {"scores": [r(s) for s in scores], "labels": [int(v) for v in yte],
            "roc": [[r(a), r(b)] for a, b in zip(fpr, tpr)],
            "pr": [[r(a), r(b)] for a, b in zip(rec, prec)],
            "thresholds": thr}


def build_comparison():
    reg = []
    dd = load_diabetes()
    Xtr, Xte, ytr, yte = train_test_split(dd.data, dd.target, test_size=0.25, random_state=RNG)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    reg_models = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=1.0),
                  "Lasso": Lasso(alpha=0.1, max_iter=5000),
                  "ElasticNet": ElasticNet(alpha=0.1, max_iter=5000),
                  "KNN": KNeighborsRegressor(10), "SVR": SVR(kernel="rbf", C=10),
                  "Bayesian": BayesianRidge(), "Huber": HuberRegressor(max_iter=500)}
    for name, m in reg_models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        reg.append({"model": name, "dataset": "Diabetes", "r2": r(r2_score(yte, p)),
                    "rmse": r(np.sqrt(mean_squared_error(yte, p)), 2),
                    "mae": r(mean_absolute_error(yte, p), 2)})
    clf = []
    dc = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(dc.data, dc.target, test_size=0.25,
                                          random_state=RNG, stratify=dc.target)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf_models = {"LDA": LinearDiscriminantAnalysis(),
                  "QDA": QuadraticDiscriminantAnalysis(reg_param=0.3),
                  "GaussianNB": GaussianNB(), "KNN": KNeighborsClassifier(10),
                  "SVM": SVC(probability=True, random_state=RNG),
                  "GBM": GradientBoostingClassifier(random_state=RNG),
                  "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                           eval_metric="logloss", random_state=RNG)}
    for name, m in clf_models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        clf.append({"model": name, "dataset": "Breast Cancer",
                    "accuracy": r(accuracy_score(yte, p)),
                    "f1": r(f1_score(yte, p)), "rocauc": r(roc_auc_score(yte, proba))})
    return {"regression": reg, "classification": clf}


WRITERS = {
    "linear": build_linear, "biasvariance": build_biasvariance,
    "regularization": build_regularization, "knnsvr": build_knnsvr,
    "bayesian": build_bayesian, "robust": build_robust, "glm": build_glm,
    "quantile": build_quantile, "discriminant": build_discriminant,
    "naivebayes": build_naivebayes, "knnsvm": build_knnsvm, "linclf": build_linclf,
    "boosting": build_boosting, "threshold": build_threshold,
    "comparison": build_comparison,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in WRITERS.items():
        data = fn()
        blob = json.dumps(data)
        (OUT / f"{name}.json").write_text(blob)
        print(f"wrote {name:15} {len(blob):>8,} bytes")
    for name in WRITERS:  # self-check: exists, valid JSON, non-empty
        obj = json.loads((OUT / f"{name}.json").read_text())
        assert obj, f"{name}.json is empty"
    print(f"OK {len(WRITERS)} artifacts -> {OUT}")


if __name__ == "__main__":
    main()
