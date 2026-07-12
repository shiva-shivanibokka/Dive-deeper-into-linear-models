"""Schema/shape tests for the precomputed web artifacts.

Each builder in ``scripts/export_web_artifacts.py`` must return a dict whose shape
matches what the corresponding frontend tab consumes. These tests fit the real
models (fast on these small datasets) and assert the contract, so a schema drift
is caught in CI instead of as a blank chart in the browser.

Run: pytest -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import export_web_artifacts as ex  # noqa: E402


def test_all_writers_registered():
    # every tab that needs an artifact has a builder
    expected = {
        "linear", "biasvariance", "regularization", "knnsvr", "bayesian", "robust",
        "glm", "quantile", "discriminant", "naivebayes", "knnsvm", "linclf",
        "boosting", "threshold", "comparison",
    }
    assert set(ex.WRITERS) == expected


def _xy_pairs(seq):
    return isinstance(seq, list) and all(len(p) == 2 for p in seq)


def test_linear():
    d = ex.build_linear()
    assert d["features"] and len(d["features"]) == 8
    for name in d["features"]:
        pf = d["perFeature"][name]
        assert _xy_pairs(pf["points"]) and _xy_pairs(pf["fit"])
        assert -1 <= pf["r2"] <= 1


def test_regularization_matrix_shape():
    d = ex.build_regularization()
    n_alpha, n_feat = len(d["alphas"]), len(d["features"])
    for key in ("ridge", "lasso", "elastic"):
        assert len(d[key]) == n_alpha
        assert all(len(row) == n_feat for row in d[key])


def test_biasvariance_degrees_monotone_range():
    d = ex.build_biasvariance()
    degs = [g["degree"] for g in d["degrees"]]
    assert degs == list(range(1, 16))
    assert all(g["trainErr"] >= 0 and g["testErr"] >= 0 for g in d["degrees"])


def test_glm_negative_prediction_signal():
    d = ex.build_glm()
    names = [m["name"] for m in d["models"]]
    assert "OLS" in names and "Poisson" in names
    frac = {m["name"]: m["fracNegative"] for m in d["models"]}
    # the whole point of the demo: GLMs never predict negative house values
    assert frac["Poisson"] == 0.0 and frac["Gamma"] == 0.0


def test_robust_has_all_estimators():
    d = ex.build_robust()
    for frame in d["contamination"]:
        for key in ("ols", "huber", "ransac", "theilsen"):
            assert _xy_pairs(frame[key])


@pytest.mark.parametrize("name", ["discriminant", "naivebayes", "knnsvm", "linclf"])
def test_boundary_grids_are_square(name):
    d = ex.WRITERS[name]()
    g = d["grid"]
    step = g["step"]

    def check(grid):
        assert len(grid) == step and all(len(row) == step for row in grid)

    if name == "discriminant":
        check(d["lda"]); check(d["qda"])
    elif name == "naivebayes":
        check(d["boundary"])
    elif name == "knnsvm":
        for e in d["knn"] + d["svm"]:
            check(e["grid"])
    else:
        for e in d["perceptronIters"]:
            check(e["grid"])
        check(d["ridge"])


def test_threshold_monotone_and_bounded():
    d = ex.build_threshold()
    assert len(d["scores"]) == len(d["labels"])
    for t in d["thresholds"]:
        assert 0 <= t["precision"] <= 1 and 0 <= t["recall"] <= 1
        assert t["tp"] + t["fn"] == d["thresholds"][0]["tp"] + d["thresholds"][0]["fn"]


def test_comparison_leaderboards():
    d = ex.build_comparison()
    assert len(d["regression"]) >= 5 and len(d["classification"]) >= 5
    for row in d["classification"]:
        assert 0 <= row["accuracy"] <= 1 and 0 <= row["rocauc"] <= 1


def test_serving_model_contract():
    d = ex.build_serving()
    n = len(d["features"])
    assert len(d["means"]) == n == len(d["stds"]) == len(d["coef"])
    assert set(d["example"]) == set(d["features"])
    assert isinstance(d["intercept"], float)
