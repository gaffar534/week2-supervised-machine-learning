"""
Week 2 - Supervised Machine Learning Models
Breast Cancer Wisconsin (Diagnostic): Logistic Regression vs Random Forest
Author: Sayyed Abdul Gaffar (24A25A0534)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.exceptions import ConvergenceWarning

RANDOM_STATE = 42

# ------------------------------------------------------------------ 1. Load
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target              # y: 0 = malignant, 1 = benign
print("Classes (0, 1):", [str(n) for n in data.target_names])

# ------------------------------------------------------------------- 2. EDA
print("\n=== EDA ===")
print("Shape (samples, features):", X.shape)
print("Data types:", X.dtypes.unique())
print("Missing values:", int(X.isnull().sum().sum()))
print("Duplicate rows:", int(X.duplicated().sum()))

counts = y.value_counts().sort_index()
for label in counts.index:
    print(f"{data.target_names[label]:>9}: {counts[label]} "
          f"({counts[label] / len(y) * 100:.2f}%)")

selected = ["mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "mean concavity", "worst area"]
print(X[selected].describe().T[["mean", "std", "min", "50%", "max"]].round(4))

col_max = X.max()
print("Largest feature maximum :", col_max.idxmax(), col_max.max())
print("Smallest feature maximum:", col_max.idxmin(), col_max.min())
print("Ratio of the two maxima : %.1f" % (col_max.max() / col_max.min()))
print("Features with max > 100 :", int((col_max > 100).sum()))
print("Features with max < 1   :", int((col_max < 1).sum()))

print("\nClass-wise means:")
for f in ["mean radius", "mean area", "mean concavity", "mean smoothness"]:
    print(f"{f:>16}: malignant={X.loc[y == 0, f].mean():.4f}  "
          f"benign={X.loc[y == 1, f].mean():.4f}")

corr = X.corrwith(y).sort_values()
print("\nStrongest correlations with target (1 = benign):")
print(corr.head(5).round(4))
print("Weakest correlations (smallest |r|):")
print(corr.abs().sort_values().head(3).round(4))
print("Correlation(mean radius, mean area): %.4f"
      % X["mean radius"].corr(X["mean area"]))

# Figure 1: class distribution
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Malignant", "Benign"], [counts[0], counts[1]], color=["#B03A2E", "#1F618D"])
for i, v in enumerate([counts[0], counts[1]]):
    ax.text(i, v + 4, f"{v} ({v / len(y) * 100:.2f}%)", ha="center")
ax.set(ylabel="Number of samples", title="Class distribution", ylim=(0, 420))
plt.tight_layout(); plt.savefig("fig_class.png", dpi=200); plt.close()

# Figure 2: histograms of two features by class
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, f in zip(axes, ["mean radius", "mean area"]):
    for c, col in [(0, "#B03A2E"), (1, "#1F618D")]:
        ax.hist(X.loc[y == c, f], bins=25, alpha=0.65, color=col,
                label=data.target_names[c])
    ax.set(xlabel=f, ylabel="Frequency", title=f"Distribution of {f}")
    ax.legend()
plt.tight_layout(); plt.savefig("fig_hist.png", dpi=200); plt.close()

# ----------------------------------------------- 3. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print("\n=== Split ===")
print("Train:", X_train.shape[0], y_train.value_counts().sort_index().to_dict())
print("Test :", X_test.shape[0], y_test.value_counts().sort_index().to_dict())
print("Train benign share: %.2f%%" % (y_train.mean() * 100))
print("Test  benign share: %.2f%%" % (y_test.mean() * 100))
mu, sd = X_train["mean area"].mean(), X_train["mean area"].std(ddof=0)
print("Training mean area: mu=%.4f sigma=%.4f (full data mean=%.4f)"
      % (mu, sd, X["mean area"].mean()))

# ---------------------------------------------------------------- 4. Models
log_reg = Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000,
                                               random_state=RANDOM_STATE))])
rand_forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

models = {"Logistic Regression": log_reg, "Random Forest": rand_forest}
preds = {}
print("\n=== Results (test set) ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds[name] = model.predict(X_test)
    acc = accuracy_score(y_test, preds[name])
    cm = confusion_matrix(y_test, preds[name])
    print(f"\n{name}")
    print("Training accuracy: %.4f" % accuracy_score(y_train, model.predict(X_train)))
    print("Test accuracy    : %.4f (%d of %d correct)"
          % (acc, (preds[name] == y_test).sum(), len(y_test)))
    print("Confusion matrix (rows = actual, columns = predicted; order: malignant, benign):\n", cm)
    print(classification_report(y_test, preds[name],
                                target_names=data.target_names, digits=4))

# Figure 3: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, p),
                           display_labels=["Malignant", "Benign"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout(); plt.savefig("fig_cm.png", dpi=200); plt.close()

# ------------------------------------------- 5. Extra analysis (supplementary)
print("\n=== Supplementary analysis ===")
wrong_lr = np.where(preds["Logistic Regression"] != y_test.values)[0]
wrong_rf = np.where(preds["Random Forest"] != y_test.values)[0]
disagree = np.where(preds["Logistic Regression"] != preds["Random Forest"])[0]
print("LR errors (test positions):", wrong_lr.tolist())
print("RF errors (test positions):", wrong_rf.tolist())
print("Models disagree on        :", disagree.tolist())

row = X_test.iloc[[0]]
logit = log_reg.decision_function(row)[0]
print("Worked example: dataset row", X_test.index[0], "actual =", y_test.iloc[0])
print("  mean area = %.1f, z-score = %.4f" % (row['mean area'].iloc[0],
                                              (row['mean area'].iloc[0] - mu) / sd))
print("  LR logit = %.4f, P(benign) = %.3e" % (logit, log_reg.predict_proba(row)[0, 1]))
print("  RF P(benign) = %.2f" % rand_forest.predict_proba(row)[0, 1])

print("\nRF top-5 importances:")
print(pd.Series(rand_forest.feature_importances_, index=X.columns).nlargest(5).round(4))
coef = pd.Series(log_reg.named_steps["clf"].coef_[0], index=X.columns)
print("LR top-5 standardised coefficients (by magnitude):")
print(coef.reindex(coef.abs().nlargest(5).index).round(4))
print("LR intercept: %.4f" % log_reg.named_steps["clf"].intercept_[0])

# Unscaled Logistic Regression, to demonstrate why scaling matters
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    lr_raw = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_raw.fit(X_train, y_train)
print("Unscaled LR test accuracy: %.4f | convergence warning raised: %s"
      % (accuracy_score(y_test, lr_raw.predict(X_test)),
         any(issubclass(w.category, ConvergenceWarning) for w in caught)))

# 5-fold stratified cross-validation (scaler re-fitted inside every fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, model in [
    ("Logistic Regression",
     Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))]:
    s = cross_val_score(model, X, y, cv=cv)
    print(f"CV {name}: folds={np.round(s, 4)} mean={s.mean():.4f} sd={s.std():.4f}")
