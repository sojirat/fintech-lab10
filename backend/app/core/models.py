from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import get_feature_frame, get_labels


@dataclass
class TrainedModels:
    iforest: Pipeline
    clf: Pipeline
    feature_names: List[str]  # after preprocessing


def _oversample(X: pd.DataFrame, y: pd.Series, target_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Random oversampling of minority class to reach target fraud ratio."""
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]

    if len(pos_idx) == 0:
        return X, y

    current_ratio = len(pos_idx) / max(1, len(y))
    if current_ratio >= target_ratio:
        return X, y

    # desired positive count:
    desired_pos = int((target_ratio / (1 - target_ratio)) * len(neg_idx))
    add_n = max(0, desired_pos - len(pos_idx))
    sampled = rng.choice(pos_idx, size=add_n, replace=True)
    keep = np.concatenate([np.arange(len(y)), sampled])

    X2 = X.iloc[keep].reset_index(drop=True)
    y2 = y.iloc[keep].reset_index(drop=True)
    return X2, y2


def train_models(tx_df: pd.DataFrame, models_dir: str) -> Dict[str, Any]:
    features = get_feature_frame(tx_df)
    y = get_labels(tx_df)

    numeric_cols = [c for c in features.columns if c not in ["Category", "Location"]]
    categorical_cols = ["Category", "Location"]

    pre_template = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    pre_iforest = clone(pre_template)
    pre_clf = clone(pre_template)

    # Isolation Forest (unsupervised)
    iforest = Pipeline(
        steps=[
            ("pre", pre_iforest),
            ("model", IsolationForest(n_estimators=200, contamination=0.01, random_state=42)),
        ]
    )

    # Supervised “shadow” model for XAI (logistic regression)
    clf = Pipeline(
        steps=[
            ("pre", pre_clf),
            ("model", LogisticRegression(max_iter=2000, class_weight=None)),
        ]
    )

    # Fit IForest on all data (unlabeled)
    iforest.fit(features)

    # Fit supervised model with imbalance handling
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None)
    X_train_os, y_train_os = _oversample(X_train, y_train, target_ratio=0.2)

    clf.fit(X_train_os, y_train_os)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred).tolist()
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    auc = None
    if y_proba is not None and y_test.nunique() > 1:
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            auc = None

    # Extract feature names for explain fallback
    # After ColumnTransformer: numeric + onehot(cat)
    ohe: OneHotEncoder = clf.named_steps["pre"].named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(categorical_cols))
    feature_names = numeric_cols + ohe_names

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(iforest, models_path / "iforest.joblib")
    joblib.dump(clf, models_path / "clf.joblib")
    joblib.dump(feature_names, models_path / "feature_names.joblib")

    return {
        "status": "trained",
        "n_rows": int(len(tx_df)),
        "label_fraud_rate": float(y.mean()) if len(y) else 0.0,
        "oversampled_fraud_rate": float(y_train_os.mean()) if len(y_train_os) else 0.0,
        "confusion_matrix": cm,
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "roc_auc": auc,
        "notes": "IsolationForest is unsupervised. LogisticRegression is used as a supervised shadow model for explanations (XAI).",
    }


def load_models(models_dir: str) -> TrainedModels:
    p = Path(models_dir)
    iforest = joblib.load(p / "iforest.joblib")
    clf = joblib.load(p / "clf.joblib")
    feature_names = joblib.load(p / "feature_names.joblib")
    return TrainedModels(iforest=iforest, clf=clf, feature_names=feature_names)


def score_transaction(models: TrainedModels, tx_row: pd.DataFrame) -> Dict[str, Any]:
    X = get_feature_frame(tx_row)

    # IsolationForest: decision_function (higher = more normal)
    if_score = float(models.iforest.named_steps["model"].decision_function(models.iforest.named_steps["pre"].transform(X))[0])

    # Convert to risk score (higher = more risky)
    risk = float(-if_score)

    proba = None
    pred = None
    try:
        proba = float(models.clf.predict_proba(X)[:, 1][0])
        pred = int(proba >= 0.5)
    except Exception:
        pred = int(models.clf.predict(X)[0])
        proba = None

    return {
        "iforest_decision_function": if_score,
        "iforest_risk_score": risk,
        "shadow_model_pred": pred,
        "shadow_model_proba": proba,
    }


def explain_transaction(models: TrainedModels, tx_row: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
    """Explain using SHAP if available, else linear contribution fallback."""
    X = get_feature_frame(tx_row)

    # Try SHAP (optional)
    try:
        import shap  # type: ignore

        # Use linear explainer for the logistic regression step
        pre = models.clf.named_steps["pre"]
        lr: LogisticRegression = models.clf.named_steps["model"]
        X_tr = pre.transform(X)

        explainer = shap.LinearExplainer(lr, pre.transform(get_feature_frame(tx_row)), feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_tr)

        # shap_values shape: (1, n_features)
        sv = np.array(shap_values).reshape(-1)
        pairs = sorted(zip(models.feature_names, sv), key=lambda t: abs(t[1]), reverse=True)[:top_k]
        return {
            "method": "shap_linear",
            "top_factors": [{"feature": f, "contribution": float(v)} for f, v in pairs],
        }
    except Exception:
        pass

    # Fallback: coefficient * transformed value
    pre = models.clf.named_steps["pre"]
    lr: LogisticRegression = models.clf.named_steps["model"]
    X_tr = pre.transform(X)
    coefs = lr.coef_.reshape(-1)

    contrib = (X_tr.toarray().reshape(-1) if hasattr(X_tr, "toarray") else np.array(X_tr).reshape(-1)) * coefs
    pairs = sorted(zip(models.feature_names, contrib), key=lambda t: abs(t[1]), reverse=True)[:top_k]

    return {
        "method": "linear_contribution",
        "top_factors": [{"feature": f, "contribution": float(v)} for f, v in pairs],
        "note": "Install SHAP for more advanced explanations. This fallback is audit-friendly and deterministic.",
    }
