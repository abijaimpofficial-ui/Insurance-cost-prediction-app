"""
Insurance Cost Prediction using Ridge and Lasso Regression
==========================================================
Steps: load data -> EDA -> preprocessing -> train/test split ->
       Linear / Ridge / Lasso (with GridSearchCV) -> metrics -> plots -> save model.
Run:   python insurance_model.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # works without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/insurance.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ 1. LOAD
if not os.path.exists(DATA_PATH):
    from generate_dataset import generate
    os.makedirs("data", exist_ok=True)
    generate().to_csv(DATA_PATH, index=False)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.describe(), "\n")

# ------------------------------------------------------------ 2. PREPROCESS
print("Missing values:\n", df.isnull().sum(), "\n")
before = len(df)
df = df.drop_duplicates()
print(f"Duplicates removed: {before - len(df)}")

# --- EDA plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["charges"], kde=True, ax=ax[0], color="steelblue")
ax[0].set_title("Distribution of Charges (right-skewed)")
sns.histplot(np.log1p(df["charges"]), kde=True, ax=ax[1], color="darkorange")
ax[1].set_title("Distribution of log(Charges)")
plt.tight_layout(); plt.savefig(f"{OUT}/01_charges_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.7)
plt.title("Age vs Charges (by smoker status)")
plt.tight_layout(); plt.savefig(f"{OUT}/02_age_vs_charges_smoker.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="smoker", y="charges", palette="Set2", hue="smoker", legend=False)
plt.title("Charges by Smoker Status")
plt.tight_layout(); plt.savefig(f"{OUT}/03_smoker_boxplot.png", dpi=150); plt.close()

# Feature / target
X = df.drop(columns="charges")
y = df["charges"]
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

# Encode + scale (scaling is REQUIRED for Ridge/Lasso penalties to be fair)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols),
])

# Correlation heatmap on numerically encoded data
enc = df.copy()
enc["sex"] = (enc["sex"] == "male").astype(int)
enc["smoker"] = (enc["smoker"] == "yes").astype(int)
enc = pd.get_dummies(enc, columns=["region"], drop_first=True, dtype=int)
plt.figure(figsize=(8, 6))
sns.heatmap(enc.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(f"{OUT}/04_correlation_heatmap.png", dpi=150); plt.close()

# ----------------------------------------------------------- 3. TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------- 4. MODELS
# Interaction (smoker x bmi) added so a linear model can capture the strong
# smoker-BMI effect. Added inside a helper so the Streamlit app reuses it.
def add_features(d):
    d = d.copy()
    d["smoker_bmi"] = (d["smoker"] == "yes").astype(int) * d["bmi"]
    d["age_sq"] = d["age"] ** 2
    return d

num_cols_fe = num_cols + ["smoker_bmi", "age_sq"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols_fe),
    ("cat", OneHotEncoder(drop="first"), cat_cols),
])

X_train_fe, X_test_fe = add_features(X_train), add_features(X_test)

models = {
    "Linear Regression": (LinearRegression(), None),
    "Ridge": (Ridge(), {"model__alpha": np.logspace(-3, 3, 30)}),
    "Lasso": (Lasso(max_iter=20000), {"model__alpha": np.logspace(-3, 3, 30)}),
}

results, fitted = {}, {}
for name, (est, grid) in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", est)])
    if grid:
        search = GridSearchCV(pipe, grid, cv=5, scoring="r2", n_jobs=-1)
        search.fit(X_train_fe, y_train)
        best = search.best_estimator_
        best_alpha = float(search.best_params_["model__alpha"])
    else:
        best = pipe.fit(X_train_fe, y_train)
        best_alpha = None
    pred = best.predict(X_test_fe)
    cv_r2 = cross_val_score(best, X_train_fe, y_train, cv=5, scoring="r2").mean()
    results[name] = {
        "best_alpha": best_alpha,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2_test": r2_score(y_test, pred),
        "R2_train": r2_score(y_train, best.predict(X_train_fe)),
        "CV_R2": float(cv_r2),
    }
    fitted[name] = (best, pred)

res_df = pd.DataFrame(results).T
print("\n===== MODEL COMPARISON =====")
print(res_df.round(4))
res_df.round(4).to_csv(f"{OUT}/metrics.csv")

# ------------------------------------------------------------ 5. PLOTS
# (a) metric comparison
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
res_df[["R2_train", "R2_test"]].plot(kind="bar", ax=ax[0], rot=15)
ax[0].set_title("R² (Train vs Test)"); ax[0].set_ylim(0, 1)
res_df[["MAE", "RMSE"]].plot(kind="bar", ax=ax[1], rot=15, color=["#4c72b0", "#dd8452"])
ax[1].set_title("Error Metrics (lower is better)")
plt.tight_layout(); plt.savefig(f"{OUT}/05_model_comparison.png", dpi=150); plt.close()

# (b) Actual vs predicted + residuals for each model
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, (name, (_, pred)) in enumerate(fitted.items()):
    axes[0, i].scatter(y_test, pred, alpha=0.6)
    lims = [y_test.min(), y_test.max()]
    axes[0, i].plot(lims, lims, "r--")
    axes[0, i].set_title(f"{name}: Actual vs Predicted")
    axes[0, i].set_xlabel("Actual"); axes[0, i].set_ylabel("Predicted")
    axes[1, i].scatter(pred, y_test - pred, alpha=0.6, color="seagreen")
    axes[1, i].axhline(0, color="r", ls="--")
    axes[1, i].set_title(f"{name}: Residuals")
    axes[1, i].set_xlabel("Predicted"); axes[1, i].set_ylabel("Residual")
plt.tight_layout(); plt.savefig(f"{OUT}/06_actual_vs_predicted_residuals.png", dpi=150); plt.close()

# (c) coefficients: shows Lasso shrinking features to zero
feat_names = (num_cols_fe +
              list(fitted["Ridge"][0].named_steps["prep"]
                   .named_transformers_["cat"].get_feature_names_out(cat_cols)))
coefs = pd.DataFrame({
    n: fitted[n][0].named_steps["model"].coef_ for n in models
}, index=feat_names)
coefs.round(2).to_csv(f"{OUT}/coefficients.csv")
coefs.plot(kind="barh", figsize=(10, 7))
plt.title("Standardised Coefficients: Linear vs Ridge vs Lasso")
plt.axvline(0, color="k", lw=0.8)
plt.tight_layout(); plt.savefig(f"{OUT}/07_coefficients.png", dpi=150); plt.close()

# (d) alpha vs CV score (Ridge / Lasso)
alphas = np.logspace(-3, 3, 30)
plt.figure(figsize=(8, 5))
for name, est in [("Ridge", Ridge()), ("Lasso", Lasso(max_iter=20000))]:
    scores = []
    for a in alphas:
        p = Pipeline([("prep", preprocessor), ("model", est.set_params(alpha=a))])
        scores.append(cross_val_score(p, X_train_fe, y_train, cv=5, scoring="r2").mean())
    plt.semilogx(alphas, scores, marker="o", ms=3, label=name)
plt.xlabel("alpha (log scale)"); plt.ylabel("CV R²")
plt.title("Effect of Regularisation Strength"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/08_alpha_vs_cv_r2.png", dpi=150); plt.close()

# ------------------------------------------------------------ 6. SAVE
best_name = res_df.loc[["Ridge", "Lasso"], "R2_test"].idxmax()   # best regularised model
print(f"\nBest model on test set: {best_name}")
joblib.dump(fitted[best_name][0], "insurance_model.pkl")
with open(f"{OUT}/summary.json", "w") as f:
    json.dump({"best_model": best_name, "results": results}, f, indent=2)
print("Saved insurance_model.pkl and plots in /outputs")
