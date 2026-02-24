from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


@dataclass
class DatasetBundle:
    tx: pd.DataFrame
    customers: pd.DataFrame
    accounts: pd.DataFrame
    merchants: pd.DataFrame
    categories: pd.DataFrame
    fraud: pd.DataFrame
    anomaly: pd.DataFrame
    suspicious_customer: pd.DataFrame


def load_dataset(data_dir: str) -> DatasetBundle:
    """Load the provided lecture dataset (folder structure)."""
    base = Path(data_dir)

    # Paths in the attached dataset
    tx_records = base / "Transaction Data" / "transaction_records.csv"
    tx_meta = base / "Transaction Data" / "transaction_metadata.csv"
    anomaly_scores = base / "Transaction Amounts" / "anomaly_scores.csv"
    fraud_ind = base / "Fraudulent Patterns" / "fraud_indicators.csv"
    suspicious = base / "Fraudulent Patterns" / "suspicious_activity.csv"
    cust = base / "Customer Profiles" / "customer_data.csv"
    account = base / "Customer Profiles" / "account_activity.csv"
    merch = base / "Merchant Information" / "merchant_data.csv"
    cat = base / "Merchant Information" / "transaction_category_labels.csv"

    missing = [str(p) for p in [tx_records, tx_meta, anomaly_scores, fraud_ind, suspicious, cust, account, merch, cat] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    df_tx = _read_csv(tx_records)
    df_meta = _read_csv(tx_meta)
    df_anom = _read_csv(anomaly_scores)
    df_fraud = _read_csv(fraud_ind)
    df_susp = _read_csv(suspicious)
    df_cust = _read_csv(cust)
    df_acc = _read_csv(account)
    df_merch = _read_csv(merch)
    df_cat = _read_csv(cat)

    # Join to build a single transaction table (tx_full)
    tx_full = df_tx.merge(df_meta, on="TransactionID", how="left") \
                  .merge(df_anom, on="TransactionID", how="left") \
                  .merge(df_fraud, on="TransactionID", how="left") \
                  .merge(df_cat, on="TransactionID", how="left") \
                  .merge(df_cust, on="CustomerID", how="left") \
                  .merge(df_acc, on="CustomerID", how="left") \
                  .merge(df_merch, on="MerchantID", how="left") \
                  .merge(df_susp, on="CustomerID", how="left")

    # Normalize datetime
    if "Timestamp" in tx_full.columns:
        tx_full["Timestamp"] = pd.to_datetime(tx_full["Timestamp"], errors="coerce")

    # Fill expected columns
    for col in ["FraudIndicator", "SuspiciousFlag"]:
        if col in tx_full.columns:
            tx_full[col] = tx_full[col].fillna(0).astype(int)

    if "AnomalyScore" in tx_full.columns:
        tx_full["AnomalyScore"] = pd.to_numeric(tx_full["AnomalyScore"], errors="coerce")

    # Sort for feature engineering (velocity)
    if "Timestamp" in tx_full.columns:
        tx_full = tx_full.sort_values("Timestamp")

    return DatasetBundle(
        tx=tx_full,
        customers=df_cust,
        accounts=df_acc,
        merchants=df_merch,
        categories=df_cat,
        fraud=df_fraud,
        anomaly=df_anom,
        suspicious_customer=df_susp,
    )
