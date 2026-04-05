# Dive Deeper into Linear Models

A comprehensive reference for regression and classification models — implemented in Python with detailed explanations, visualizations, and real-world datasets.

---

## Repository Structure

```
Dive-deeper-into-linear-models/
├── linear_regression_models.ipynb      ← All regression models
├── logistic_regression_models.ipynb    ← All classification models
└── README.md                           ← This file
```

---

## Notebooks at a Glance

### `linear_regression_models.ipynb`
**Dataset:** California Housing Dataset (20,640 samples, 8 features)  
**Task:** Predict median house values across California block groups

| # | Model | Key Concept |
|---|-------|-------------|
| 1 | Multiple Linear Regression | Best-fit hyperplane through multi-dimensional data |
| 2 | Polynomial Regression | Captures non-linear (curved) relationships |
| 3 | Ridge Regression (L2) | Shrinks coefficients to reduce overfitting; keeps all features |
| 4 | Lasso Regression (L1) | Zeros out irrelevant features; automatic feature selection |
| 5 | ElasticNet Regression | Combination of Ridge + Lasso penalties |
| 6 | KNN Regression | Predicts based on average of K nearest neighbors |
| 7 | Bayesian Linear Regression | Probabilistic predictions with uncertainty quantification |
| 8 | Support Vector Regression (SVR) | Fits a tube around the data; robust to outliers |

---

### `logistic_regression_models.ipynb`
**Datasets:**
- **Breast Cancer Wisconsin** (569 samples, 30 features) — binary classification
- **Wine Dataset** (178 samples, 13 features) — multi-class classification

| # | Model | Dataset | Key Concept |
|---|-------|---------|-------------|
| 1 | Linear Discriminant Analysis (LDA) | Wine | Finds the best linear projection to separate classes |
| 2 | Quadratic Discriminant Analysis (QDA) | Wine | Like LDA but with flexible, curved boundaries |
| 3 | Gaussian Naive Bayes | Breast Cancer | Probabilistic classification using Bayes' theorem |
| 4 | K-Nearest Neighbors (KNN) | Wine | Majority vote from K nearest neighbors |
| 5 | Support Vector Machine (SVM) | Breast Cancer | Maximum-margin hyperplane; kernel trick for non-linearity |
| 6 | Gradient Boosting (GBM) | Breast Cancer | Sequential trees that correct each other's errors |
| 7 | XGBoost | Breast Cancer | High-performance gradient boosting with regularization |

---

## Key Concepts Explained

### What Is Supervised Learning?

Supervised learning is when a model **learns from labeled examples**. Each training example has both an **input** (features) and a known **output** (label). The model finds the relationship between inputs and outputs, then applies it to new, unseen data.

Two main types:
- **Regression** → predict a continuous number (house price, temperature)
- **Classification** → predict a category (cancer or not, which wine type)

---

### The Bias-Variance Trade-off

One of the most important concepts in machine learning:

- **High Bias (Underfitting)** → The model is too simple. It misses patterns even in training data. Example: fitting a straight line to data that curves.
- **High Variance (Overfitting)** → The model is too complex. It memorizes training data but fails on new data. Example: a 10th-degree polynomial that wiggles through every point.
- **Sweet spot** → A model that generalizes well — not too simple, not too complex.

Ridge, Lasso, and ElasticNet are all designed to manage this trade-off through **regularization**.

---

### Regularization — Why It Matters

Plain linear regression finds the coefficients that minimize error on training data. But with many features, this can produce huge, unstable coefficients that overfit.

**Regularization adds a penalty** for large coefficients:

| Method | Penalty | Effect |
|--------|---------|--------|
| Ridge (L2) | Sum of squared coefficients | Shrinks all coefficients smoothly |
| Lasso (L1) | Sum of absolute coefficients | Zeros out some coefficients entirely |
| ElasticNet | Mix of L1 + L2 | Balanced shrinkage + feature selection |

---

### Evaluation Metrics

#### Regression Metrics

| Metric | Formula (simplified) | What It Means |
|--------|---------------------|---------------|
| MAE | Average of absolute errors | Your average prediction error in original units |
| RMSE | Square root of average squared errors | Like MAE but penalizes large errors more |
| R² Score | 1 - (residual variance / total variance) | How much of the variation your model explains (1.0 = perfect) |

#### Classification Metrics

| Metric | What It Means |
|--------|---------------|
| Accuracy | % of predictions that are correct |
| Precision | Of predicted positives, how many are truly positive? |
| Recall | Of actual positives, how many did we correctly find? |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | How well the model separates the two classes across all thresholds |

> **Important:** Accuracy can be misleading on imbalanced datasets. Always examine precision, recall, and the confusion matrix.

---

## Model Selection Guide

### For Regression

| Situation | Recommended Model |
|-----------|------------------|
| Quick baseline | Multiple Linear Regression |
| Many correlated features | Ridge Regression |
| Many irrelevant features | Lasso Regression |
| Uncertain which features matter | ElasticNet |
| Non-linear patterns | Polynomial Regression or SVR |
| No assumptions needed | KNN Regression |
| Need uncertainty estimates | Bayesian Linear Regression |
| Robust to outliers needed | SVR |

### For Classification

| Situation | Recommended Model |
|-----------|------------------|
| Quick baseline | Gaussian Naive Bayes |
| Multi-class + visualization | LDA |
| Different class spreads | QDA |
| No distributional assumptions | KNN |
| High-dimensional data | SVM |
| Maximum accuracy needed | XGBoost or GBM |
| Interpretable + feature importance | Gradient Boosting |

---

## Datasets Used

### California Housing Dataset
- **Source:** 1990 US Census (via `sklearn.datasets.fetch_california_housing`)
- **Samples:** 20,640
- **Features:** 8 (income, house age, rooms, bedrooms, population, occupancy, lat/lon)
- **Target:** Median house value (in $100,000s)

### Breast Cancer Wisconsin Dataset
- **Source:** UCI Machine Learning Repository (via `sklearn.datasets.load_breast_cancer`)
- **Samples:** 569
- **Features:** 30 (measurements of cell nuclei from biopsy images)
- **Target:** Malignant (0) or Benign (1)

### Wine Dataset
- **Source:** UCI Machine Learning Repository (via `sklearn.datasets.load_wine`)
- **Samples:** 178
- **Features:** 13 (chemical properties of wine)
- **Target:** Wine cultivar class (0, 1, or 2)

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shiva-shivanibokka/Dive-deeper-into-linear-models.git
cd Dive-deeper-into-linear-models
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Then open either notebook and run cells from top to bottom.

These notebooks can also be run directly in [Google Colab](https://colab.research.google.com/) — no local setup required.

---

## Dependencies

| Library | Version (recommended) | Purpose |
|---------|----------------------|---------|
| Python | 3.8+ | Language |
| NumPy | 1.21+ | Numerical computing |
| Pandas | 1.3+ | Data manipulation |
| Matplotlib | 3.4+ | Visualization |
| Seaborn | 0.11+ | Statistical visualization |
| scikit-learn | 1.0+ | All ML models |
| XGBoost | 1.5+ | XGBoost model |
| Jupyter | 6.0+ | Notebook interface |

---

## What's Not in This Repo

The following are intentionally excluded and covered in a **separate repository**:

- Random Forest
- Bagging / Bootstrap Aggregating
- Extra Trees
- AdaBoost
- Stacking / Blending

This keeps the focus clean: **linear and kernel-based models first**, ensemble methods second.

---

## Author

**Shivani Bokka**

This repository was built as both a teaching resource and a professional portfolio piece demonstrating practical knowledge of supervised machine learning.

---

## License

This project is open-source and available for educational use.
