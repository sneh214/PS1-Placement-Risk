"""
PS1 - Placement Risk Modeling
Step 2 & 3: Feature engineering + Train XGBoost models
"""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import shap

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../data")
df = pd.read_csv("student_data.csv")
print(f"Loaded {len(df)} records")

# ── Feature engineering ────────────────────────────────────────────────────
df["employability_score"] = (
    0.35 * (df["cgpa"] / 10.0) +
    0.25 * (df["internships"] / 3.0).clip(0, 1) +
    0.20 * df["institute_score"] +
    0.15 * df["industry_demand"] +
    0.05 * (df["certifications"] / 5.0).clip(0, 1)
)

df["engagement_score"] = (
    0.6 * df["job_portal_active"] +
    0.4 * (df["certifications"] / 5.0).clip(0, 1)
)

le_course = LabelEncoder()
df["course_enc"] = le_course.fit_transform(df["course"])

FEATURES = [
    "cgpa", "internships", "certifications", "job_portal_active",
    "institute_tier", "industry_demand", "institute_score",
    "employability_score", "engagement_score", "course_enc"
]

LABEL_ORDER = {"placed_3m": 0, "placed_6m": 1, "placed_12m": 2, "delayed": 3}
df["label_enc"] = df["placement_label"].map(LABEL_ORDER)

X = df[FEATURES]
y_cls = df["label_enc"]
y_sal = df["starting_salary"]

X_train, X_test, y_cls_train, y_cls_test, y_sal_train, y_sal_test = train_test_split(
    X, y_cls, y_sal, test_size=0.2, random_state=42, stratify=y_cls
)

# ── Train placement classifier ─────────────────────────────────────────────
clf = XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42
)
clf.fit(X_train, y_cls_train,
        eval_set=[(X_test, y_cls_test)], verbose=False)

y_pred = clf.predict(X_test)
labels = ["placed_3m", "placed_6m", "placed_12m", "delayed"]
print("\n── Placement Classifier ──")
print(classification_report(y_cls_test, y_pred, target_names=labels))

# ── Train salary regressor ─────────────────────────────────────────────────
reg = XGBRegressor(
    n_estimators=150, max_depth=4, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, random_state=42
)
reg.fit(X_train, y_sal_train,
        eval_set=[(X_test, y_sal_test)], verbose=False)

sal_pred = reg.predict(X_test)
mae = mean_absolute_error(y_sal_test, sal_pred)
print(f"\n── Salary Regressor ──")
print(f"MAE: Rs {mae:,.0f}")

# ── SHAP explainer ─────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test[:100])
print("\nSHAP explainer ready.")

# ── Save models ────────────────────────────────────────────────────────────
os.makedirs("../models/saved", exist_ok=True)
with open("../models/saved/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("../models/saved/regressor.pkl", "wb") as f:
    pickle.dump(reg, f)
with open("../models/saved/explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)
with open("../models/saved/label_encoder.pkl", "wb") as f:
    pickle.dump(le_course, f)
with open("../models/saved/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

df.to_csv("student_data_features.csv", index=False)
print("\nAll models saved to models/saved/")
