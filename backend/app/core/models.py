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
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import get_feature_frame, get_labels


@dataclass
class TrainedModels:
    iforest: Pipeline
    clf: Pipeline
    feature_names: List[str]  # after preprocessing


def _balance_dataset(X: pd.DataFrame, y: pd.Series, target_ratio: float = 0.5, random_state: int = 42, use_smote: bool = True, strategy: str = 'hybrid') -> Tuple[pd.DataFrame, pd.Series]:
    """Balance dataset using oversampling, undersampling, or hybrid approach.

    Args:
        X: Features
        y: Labels
        target_ratio: Target ratio of minority class (0.5 = balanced)
        random_state: Random seed
        use_smote: Whether to use SMOTE for oversampling
        strategy: 'oversample', 'undersample', or 'hybrid' (default)

    Returns:
        Balanced X, y
    """
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]

    if len(pos_idx) == 0:
        return X, y

    rng = np.random.default_rng(random_state)
    current_pos = len(pos_idx)
    current_neg = len(neg_idx)

    if strategy == 'undersample':
        # Undersample majority class to match minority
        # Keep all fraud cases, randomly sample non-fraud
        desired_neg = int(current_pos / target_ratio - current_pos)
        desired_neg = max(desired_neg, current_pos)  # At least equal to fraud

        if desired_neg < current_neg:
            sampled_neg = rng.choice(neg_idx, size=desired_neg, replace=False)
            keep_idx = np.concatenate([pos_idx, sampled_neg])
            X_res = X.iloc[keep_idx].reset_index(drop=True)
            y_res = y.iloc[keep_idx].reset_index(drop=True)
            return X_res, y_res
        else:
            return X, y

    elif strategy == 'hybrid':
        # Hybrid: undersample majority moderately + oversample minority
        # This gives better results than pure oversampling

        # Step 1: Undersample majority to reduce extreme imbalance
        # Target: reduce to 3x minority (e.g., fraud:non-fraud = 1:3)
        intermediate_neg = min(current_neg, current_pos * 3)
        if intermediate_neg < current_neg:
            sampled_neg = rng.choice(neg_idx, size=intermediate_neg, replace=False)
        else:
            sampled_neg = neg_idx

        # Combine fraud + undersampled non-fraud
        keep_idx = np.concatenate([pos_idx, sampled_neg])
        X_intermediate = X.iloc[keep_idx].reset_index(drop=True)
        y_intermediate = y.iloc[keep_idx].reset_index(drop=True)

        # Step 2: Oversample minority to reach target ratio
        pos_intermediate = np.where(y_intermediate.values == 1)[0]
        neg_intermediate = np.where(y_intermediate.values == 0)[0]

        desired_pos = int((target_ratio / (1 - target_ratio)) * len(neg_intermediate))

        if desired_pos > len(pos_intermediate) and HAS_SMOTE and use_smote and len(pos_intermediate) >= 2:
            try:
                smote = SMOTE(
                    sampling_strategy={1: desired_pos},
                    random_state=random_state,
                    k_neighbors=min(5, len(pos_intermediate) - 1)
                )
                X_array = X_intermediate.values if hasattr(X_intermediate, 'values') else X_intermediate
                X_res, y_res = smote.fit_resample(X_array, y_intermediate)
                return pd.DataFrame(X_res, columns=X_intermediate.columns if hasattr(X_intermediate, 'columns') else None), pd.Series(y_res)
            except Exception as e:
                print(f"SMOTE failed in hybrid strategy, using random oversampling: {e}")

        # Fallback: random oversampling
        if desired_pos > len(pos_intermediate):
            add_n = desired_pos - len(pos_intermediate)
            sampled = rng.choice(pos_intermediate, size=add_n, replace=True)
            keep = np.concatenate([np.arange(len(y_intermediate)), sampled])
            X_res = X_intermediate.iloc[keep].reset_index(drop=True)
            y_res = y_intermediate.iloc[keep].reset_index(drop=True)
            return X_res, y_res

        return X_intermediate, y_intermediate

    else:  # 'oversample' (original behavior)
        # Use SMOTE if available and requested
        if HAS_SMOTE and use_smote and len(pos_idx) >= 2:
            try:
                desired_pos = int((target_ratio / (1 - target_ratio)) * len(neg_idx))

                if desired_pos > len(pos_idx):
                    smote = SMOTE(
                        sampling_strategy={1: desired_pos},
                        random_state=random_state,
                        k_neighbors=min(5, len(pos_idx) - 1)
                    )
                    X_array = X.values if hasattr(X, 'values') else X
                    X_res, y_res = smote.fit_resample(X_array, y)
                    return pd.DataFrame(X_res, columns=X.columns if hasattr(X, 'columns') else None), pd.Series(y_res)
            except Exception as e:
                print(f"SMOTE failed, falling back to random oversampling: {e}")

        # Random oversampling fallback
        current_ratio = len(pos_idx) / max(1, len(y))
        if current_ratio >= target_ratio:
            return X, y

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

    # Ensemble of RandomForest and GradientBoosting for better performance
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )

    # Voting classifier combines both models
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1.2, 0.8]  # Slightly favor RandomForest
    )

    clf = Pipeline(
        steps=[
            ("pre", pre_clf),
            ("model", ensemble),
        ]
    )

    # Fit IForest on all data (unlabeled)
    iforest.fit(features)

    # Use Stratified K-Fold for better evaluation with imbalanced data
    # This ensures each fold has representative fraud samples
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store fold results
    fold_cms = []
    fold_prs = []
    fold_rcs = []
    fold_f1s = []
    fold_aucs = []

    # Train final model on 80% with hybrid balancing
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.20, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )
    X_train_balanced, y_train_balanced = _balance_dataset(
        X_train, y_train, target_ratio=0.5, use_smote=HAS_SMOTE, strategy='hybrid'
    )

    # Fit final model
    clf.fit(X_train_balanced, y_train_balanced)

    # Evaluate using K-Fold CV on training set for more stable metrics
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Balance training fold using hybrid strategy
        X_fold_train_balanced, y_fold_train_balanced = _balance_dataset(
            X_fold_train, y_fold_train, target_ratio=0.5, use_smote=HAS_SMOTE, strategy='hybrid'
        )

        # Clone and train fold model
        fold_clf = clone(clf)
        fold_clf.fit(X_fold_train_balanced, y_fold_train_balanced)

        # Evaluate fold with optimized threshold
        y_fold_proba = None
        try:
            y_fold_proba = fold_clf.predict_proba(X_fold_val)[:, 1]
            # Use threshold 0.30 for evaluation (balanced for production)
            y_fold_pred = (y_fold_proba >= 0.30).astype(int)
        except Exception:
            y_fold_pred = fold_clf.predict(X_fold_val)

        fold_cm = confusion_matrix(y_fold_val, y_fold_pred)
        fold_pr, fold_rc, fold_f1, _ = precision_recall_fscore_support(
            y_fold_val, y_fold_pred, average="binary", zero_division=0
        )

        fold_auc = None
        if y_fold_proba is not None and y_fold_val.nunique() > 1:
            try:
                fold_auc = float(roc_auc_score(y_fold_val, y_fold_proba))
            except Exception:
                pass

        fold_cms.append(fold_cm)
        fold_prs.append(fold_pr)
        fold_rcs.append(fold_rc)
        fold_f1s.append(fold_f1)
        if fold_auc is not None:
            fold_aucs.append(fold_auc)

    # Calculate mean metrics across folds
    pr = float(np.mean(fold_prs))
    rc = float(np.mean(fold_rcs))
    f1 = float(np.mean(fold_f1s))
    auc = float(np.mean(fold_aucs)) if fold_aucs else None

    # For confusion matrix, sum all folds
    cm_sum = np.sum(fold_cms, axis=0)
    cm = cm_sum.tolist()

    # Also evaluate on held-out test set for reference (with threshold 0.30)
    y_proba_test = None
    try:
        y_proba_test = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= 0.30).astype(int)
    except Exception:
        y_pred_test = clf.predict(X_test)

    cm_test = confusion_matrix(y_test, y_pred_test).tolist()
    pr_test, rc_test, f1_test, _ = precision_recall_fscore_support(y_test, y_pred_test, average="binary", zero_division=0)

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
        "balanced_fraud_rate": float(y_train_balanced.mean()) if len(y_train_balanced) else 0.0,
        "train_samples": int(len(y_train_balanced)),
        "cv_folds": n_splits,
        "cv_confusion_matrix": cm,
        "cv_precision": pr,
        "cv_recall": rc,
        "cv_f1": f1,
        "cv_roc_auc": auc,
        "test_confusion_matrix": cm_test,
        "test_precision": float(pr_test),
        "test_recall": float(rc_test),
        "test_f1": float(f1_test),
        "notes": f"IsolationForest (unsupervised). Ensemble (RF+GB) with hybrid balancing (undersample majority + {'SMOTE' if HAS_SMOTE else 'random'} oversample minority). Using {n_splits}-Fold Stratified CV with threshold=0.30 for balanced evaluation.",
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
        # Use optimized threshold for fraud detection (balance precision/recall)
        pred = int(proba >= 0.15)
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

    # Fallback: use feature importances for tree-based models
    pre = models.clf.named_steps["pre"]
    model = models.clf.named_steps["model"]
    X_tr = pre.transform(X)

    # For tree-based models, use feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        X_vals = X_tr.toarray().reshape(-1) if hasattr(X_tr, "toarray") else np.array(X_tr).reshape(-1)
        contrib = X_vals * importances
        pairs = sorted(zip(models.feature_names, contrib), key=lambda t: abs(t[1]), reverse=True)[:top_k]

        return {
            "method": "feature_importance_contribution",
            "top_factors": [{"feature": f, "contribution": float(v)} for f, v in pairs],
            "note": "Using feature importances from ensemble model. Install SHAP for more detailed explanations.",
        }

    # Ultimate fallback for non-tree models
    X_vals = X_tr.toarray().reshape(-1) if hasattr(X_tr, "toarray") else np.array(X_tr).reshape(-1)
    pairs = sorted(zip(models.feature_names, X_vals), key=lambda t: abs(t[1]), reverse=True)[:top_k]

    return {
        "method": "value_magnitude",
        "top_factors": [{"feature": f, "contribution": float(v)} for f, v in pairs],
        "note": "Basic feature value magnitude ranking.",
    }
