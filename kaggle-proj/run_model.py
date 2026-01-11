import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

# =========================
# Load data
# =========================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

train = train.drop(columns=["id"])
test = test.drop(columns=["id"])

X = train.drop(columns=["target"]).fillna(train.median())
y = train["target"]
X_test = test.fillna(train.median())

num_classes = y.nunique()

# =========================
# FINAL ENSEMBLE CONFIG
# =========================
configs = [
    # diversity models
    {"max_features": 0.3, "n_estimators": 900,  "weight": 0.9, "seed": 42},
    {"max_features": 0.5, "n_estimators": 800,  "weight": 0.9, "seed": 52},

    # strong core models
    {"max_features": 0.7, "n_estimators": 1100, "weight": 2.2, "seed": 62},
    {"max_features": 1.0, "n_estimators": 1300, "weight": 2.4, "seed": 72},

    # extra strong replica (same bias, different randomness)
    {"max_features": 1.0, "n_estimators": 1300, "weight": 1.6, "seed": 99},
]

test_proba_sum = np.zeros((len(X_test), num_classes))
total_weight = 0.0

print("Training FINAL ExtraTrees ensemble...")

for cfg in configs:
    print(
        f"max_features={cfg['max_features']} | "
        f"trees={cfg['n_estimators']} | "
        f"weight={cfg['weight']}"
    )

    model = ExtraTreesClassifier(
        n_estimators=cfg["n_estimators"],
        max_features=cfg["max_features"],
        min_samples_leaf=1,
        random_state=cfg["seed"],
        n_jobs=-1
    )

    model.fit(X, y)
    proba = model.predict_proba(X_test)

    test_proba_sum += cfg["weight"] * proba
    total_weight += cfg["weight"]

# =========================
# Final prediction
# =========================
avg_test_proba = test_proba_sum / total_weight
final_preds = np.argmax(avg_test_proba, axis=1)

# =========================
# Submission
# =========================
submission = sample_sub.copy()
submission["target"] = final_preds.astype(int)
submission.to_csv("submission_FINAL_LOCK.csv", index=False)

print("submission_FINAL_LOCK.csv ready 🚀")