// Single source of truth for the demo tabs. The tab bar, panel header, and the
// lazy-loaded component map in page.tsx all derive from this list.
export type ModelTab = {
  id: string;
  group: "Regression" | "Classification" | "Capstone";
  title: string;
  tagline: string;
  dataset: string;
  help: string; // shown in the panel's "?" tooltip
};

export const MODEL_TABS: ModelTab[] = [
  {
    id: "linear", group: "Regression", title: "Linear Regression", dataset: "California Housing",
    tagline: "Pick a feature and watch the ordinary least-squares line of best fit, with its R² and RMSE.",
    help: "Linear regression fits a straight line that minimizes the sum of squared vertical distances to the points. Each feature is fit on its own here so you can see how strongly it alone predicts house value.",
  },
  {
    id: "biasvariance", group: "Regression", title: "Bias–Variance", dataset: "Diabetes (BMI)",
    tagline: "Drag the polynomial degree from 1 to 15 and watch the fit go from underfitting to overfitting as train and test error diverge.",
    help: "A low-degree curve is too rigid (high bias, underfits); a high-degree curve wiggles through every point (high variance, overfits). The gap between the train and test error curves is the tell-tale sign of overfitting.",
  },
  {
    id: "regularization", group: "Regression", title: "Regularization", dataset: "Diabetes",
    tagline: "Slide the penalty α and watch Ridge, Lasso, and ElasticNet shrink coefficients — Lasso drives some to exactly zero.",
    help: "Regularization adds a penalty on coefficient size. Ridge (L2) shrinks all coefficients smoothly; Lasso (L1) zeroes some out entirely (feature selection); ElasticNet blends both.",
  },
  {
    id: "knnsvr", group: "Regression", title: "KNN & SVR", dataset: "California Housing",
    tagline: "Change k for nearest-neighbor regression, or the ε-tube and C for support-vector regression, and watch the fit respond.",
    help: "KNN regression averages the k nearest points — small k is jagged, large k is smooth. SVR fits a tube of width ε and only penalizes points outside it, which makes it robust to small errors.",
  },
  {
    id: "bayesian", group: "Regression", title: "Bayesian Regression", dataset: "Diabetes (BMI)",
    tagline: "Increase the prior strength λ and watch the fit shrink toward zero while the uncertainty band widens away from the data.",
    help: "Bayesian regression returns a distribution over fits, not one line. A stronger prior pulls the mean toward zero and produces a credible band that is narrow where data is dense and wide where it is sparse.",
  },
  {
    id: "robust", group: "Regression", title: "Robust Regression", dataset: "Diabetes (BMI)",
    tagline: "Inject outliers and watch ordinary least squares swing toward them while Huber, RANSAC, and Theil-Sen barely move.",
    help: "Squared loss lets a few extreme points dominate OLS. Huber caps the loss on large residuals, RANSAC fits on an inlier consensus, and Theil-Sen uses the median of pairwise slopes — all resist outliers.",
  },
  {
    id: "glm", group: "Regression", title: "Generalized Linear Models", dataset: "California Housing",
    tagline: "Switch between OLS, Poisson, Gamma, and Tweedie and see that OLS predicts impossible negative house values while the GLMs never do.",
    help: "GLMs pair a linear predictor with a link function and a target distribution suited to positive or skewed data. Unlike OLS, they keep predictions in a valid range (e.g. strictly positive).",
  },
  {
    id: "quantile", group: "Regression", title: "Quantile Regression", dataset: "Diabetes (BMI)",
    tagline: "Slide across quantiles from the 10th to the 90th percentile to build a full prediction interval instead of a single mean estimate.",
    help: "Ordinary regression predicts the mean. Quantile regression uses the pinball loss to predict a chosen percentile, so stacking several gives you a prediction interval rather than one number.",
  },
  {
    id: "discriminant", group: "Classification", title: "Discriminant Analysis", dataset: "Wine (2 features)",
    tagline: "Toggle LDA and QDA to compare a shared linear boundary against per-class curved boundaries, with covariance ellipses drawn in.",
    help: "LDA assumes every class shares one covariance, giving straight boundaries. QDA lets each class have its own covariance, giving curved boundaries — more flexible, but needs more data per class.",
  },
  {
    id: "naivebayes", group: "Classification", title: "Naive Bayes", dataset: "Iris (2 features)",
    tagline: "See how Gaussian Naive Bayes carves up the feature space assuming each feature is independent within a class.",
    help: "Naive Bayes applies Bayes' theorem while pretending features are independent given the class. Despite that strong assumption it is fast and often a strong baseline.",
  },
  {
    id: "knnsvm", group: "Classification", title: "KNN & SVM", dataset: "Iris (2 features)",
    tagline: "Change k, or the SVM kernel, and watch the decision boundary reshape from local votes to maximum-margin curves.",
    help: "KNN classifies by majority vote among the k nearest points. SVM finds the widest separating margin and can bend it with a kernel (linear, RBF, or polynomial).",
  },
  {
    id: "linclf", group: "Classification", title: "Linear Classifiers", dataset: "Iris (2 features)",
    tagline: "Step the Perceptron through training iterations to watch its boundary settle, next to the deterministic Ridge Classifier.",
    help: "The Perceptron learns online, nudging its boundary after each mistake until the classes separate. The Ridge Classifier solves a regularized least-squares problem in one shot — no iteration.",
  },
  {
    id: "boosting", group: "Classification", title: "Boosting & Importance", dataset: "Breast Cancer",
    tagline: "Compare Gradient Boosting and XGBoost feature importances, and trace how the top feature moves the prediction.",
    help: "Boosting adds many small trees, each correcting the last one's errors. Feature importance shows which measurements drive the model; partial dependence shows the average effect of one feature.",
  },
  {
    id: "threshold", group: "Classification", title: "Threshold, ROC & Metrics", dataset: "Breast Cancer",
    tagline: "Drag the decision threshold and watch the confusion matrix, precision, recall, F1, and the ROC and PR operating points move live.",
    help: "A classifier outputs a probability; the threshold turns it into a yes/no. Lowering it catches more positives (higher recall) at the cost of more false alarms (lower precision).",
  },
  {
    id: "comparison", group: "Capstone", title: "Model Comparison", dataset: "Diabetes + Breast Cancer",
    tagline: "A sortable leaderboard scoring every regression and classification model on the same held-out data.",
    help: "Every model in this playground, benchmarked on a shared train/test split. Sort by any metric to see which family wins where — and that no single model dominates everything.",
  },
  {
    id: "serving", group: "Capstone", title: "Live API", dataset: "served via /api/predict",
    tagline: "Edit a house's features and hit predict — a real serverless function runs a trained Ridge model live, not a precomputed lookup.",
    help: "This tab makes an actual network request to a serverless function. The Ridge model is trained offline in Python and served as coefficients; the function standardizes your inputs and computes the prediction on the fly.",
  },
  {
    id: "about", group: "Capstone", title: "About", dataset: "the whole series",
    tagline: "A plain-language guide to every model here, how each demo works, and how it all fits together.",
    help: "What every model is, what each demo shows, and how the precomputed-artifact approach keeps it all running in your browser.",
  },
];
