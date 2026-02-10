from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "Amount",
    "AnomalyScore",
    "Age",
    "AccountBalance",
]

CATEGORICAL_COLS = [
    "Category",
    "Location",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering aligned with Lecture 10 (velocity/ratio/time)."""
    out = df.copy()

    # hour of day
    if "Timestamp" in out.columns:
        out["Hour"] = out["Timestamp"].dt.hour.fillna(0).astype(int)
    else:
        out["Hour"] = 0

    # days since last login
    if "LastLogin" in out.columns:
        out["LastLogin"] = pd.to_datetime(out["LastLogin"], errors="coerce")
        if "Timestamp" in out.columns:
            out["DaysSinceLastLogin"] = (out["Timestamp"] - out["LastLogin"]).dt.total_seconds() / 86400.0
        else:
            out["DaysSinceLastLogin"] = np.nan
    else:
        out["DaysSinceLastLogin"] = np.nan

    # velocity feature: transactions per customer per hour bucket
    if "Timestamp" in out.columns:
        hour_bucket = out["Timestamp"].dt.floor("H")
        out["_hour_bucket"] = hour_bucket
        out["TxCount_1h"] = out.groupby(["CustomerID", "_hour_bucket"])["TransactionID"].transform("count")
        out.drop(columns=["_hour_bucket"], inplace=True, errors="ignore")
    else:
        out["TxCount_1h"] = 1

    # ratio: amount / customer average
    out["CustomerAvgAmount"] = out.groupby("CustomerID")["Amount"].transform("mean")
    out["AmountToCustomerAvg"] = out["Amount"] / out["CustomerAvgAmount"].replace({0: np.nan})

    # ensure all numeric columns exist
    for col in ["AnomalyScore", "AccountBalance"]:
        if col not in out.columns:
            out[col] = np.nan

    # basic clean
    for c in ["Amount", "AnomalyScore", "Age", "AccountBalance", "Hour", "DaysSinceLastLogin", "TxCount_1h", "AmountToCustomerAvg"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the final feature frame used by models."""
    d = add_derived_features(df)

    # Add additional numeric cols used by models
    numeric = NUMERIC_COLS + ["Hour", "DaysSinceLastLogin", "TxCount_1h", "AmountToCustomerAvg"]
    categorical = CATEGORICAL_COLS

    # Some datasets may miss categoricals
    for c in categorical:
        if c not in d.columns:
            d[c] = "unknown"
        d[c] = d[c].fillna("unknown").astype(str)

    # Fill numeric
    for c in numeric:
        if c not in d.columns:
            d[c] = np.nan

    return d[numeric + categorical].copy()


def get_labels(df: pd.DataFrame) -> pd.Series:
    if "FraudIndicator" not in df.columns:
        return pd.Series([0] * len(df))
    return df["FraudIndicator"].astype(int)
