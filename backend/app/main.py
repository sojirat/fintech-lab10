from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core.data_loader import load_dataset
from .core.models import train_models, load_models, score_transaction, explain_transaction
from .core.aml import detect_smurfing, detect_cycles, graph_summary
from .core.cases import save_case, list_cases, read_case
try:
    from .core.rate_limit import RateLimitMiddleware
except Exception:  # optional dependency in local tests
    RateLimitMiddleware = None  # type: ignore


DATA_DIR = os.getenv("DATA_DIR", "/app/data")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
DB_DIR = os.getenv("DB_DIR", "/app/db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app = FastAPI(
    title="FinTech Lab 10 API — Fraud + AML Graph + XAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if RateLimitMiddleware is not None and os.getenv("DISABLE_RATE_LIMIT", "0") != "1":
    app.add_middleware(RateLimitMiddleware, redis_url=REDIS_URL, limit_per_min=RATE_LIMIT_PER_MIN)


class ScoreRequest(BaseModel):
    transaction_id: int


class ExplainRequest(BaseModel):
    transaction_id: int
    top_k: int = 10


class CaseRequest(BaseModel):
    transaction_id: int
    analyst: Optional[str] = None
    notes: Optional[str] = None


# In-memory dataset cache
DATA: Optional[pd.DataFrame] = None


def get_data() -> pd.DataFrame:
    global DATA
    if DATA is None:
        bundle = load_dataset(DATA_DIR)
        DATA = bundle.tx
    return DATA


def get_tx_row(tx_id: int) -> pd.DataFrame:
    df = get_data()
    r = df[df["TransactionID"] == tx_id]
    if r.empty:
        raise HTTPException(status_code=404, detail=f"TransactionID {tx_id} not found")
    return r.head(1).copy()


def try_load_models():
    try:
        return load_models(MODELS_DIR)
    except Exception:
        raise HTTPException(status_code=409, detail="Models not trained yet. Run POST /fraud/train first.")


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "data_loaded": DATA is not None,
        "data_dir": DATA_DIR,
        "models_dir": MODELS_DIR,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/dataset/reload")
def dataset_reload():
    global DATA
    bundle = load_dataset(DATA_DIR)
    DATA = bundle.tx
    return {"status": "reloaded", "rows": int(len(DATA))}


@app.get("/dataset/summary")
def dataset_summary():
    df = get_data()

    fraud_rate = float(df["FraudIndicator"].mean()) if "FraudIndicator" in df.columns else 0.0
    suspicious_rate = float(df["SuspiciousFlag"].mean()) if "SuspiciousFlag" in df.columns else 0.0

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "fraud_rate": fraud_rate,
        "suspicious_customer_rate": suspicious_rate,
        "amount_min": float(df["Amount"].min()) if "Amount" in df.columns else None,
        "amount_max": float(df["Amount"].max()) if "Amount" in df.columns else None,
    }


@app.get("/transactions/{tx_id}")
def get_transaction(tx_id: int):
    r = get_tx_row(tx_id)
    rec = r.iloc[0].to_dict()
    # convert Timestamp to string
    if "Timestamp" in rec and isinstance(rec["Timestamp"], pd.Timestamp):
        rec["Timestamp"] = rec["Timestamp"].isoformat()
    return rec


@app.post("/fraud/train")
def fraud_train():
    df = get_data()
    return train_models(df, MODELS_DIR)


@app.get("/fraud/top")
def fraud_top(limit: int = 30):
    models = try_load_models()
    df = get_data()

    # score in batch on a sample to keep response quick
    sample = df.head(1000).copy() if len(df) > 1000 else df.copy()

    # compute risk score quickly
    X = sample  # will be converted inside score_transaction row-by-row
    risks = []
    for tx_id in sample["TransactionID"].astype(int).tolist():
        row = get_tx_row(int(tx_id))
        s = score_transaction(models, row)
        risks.append((int(tx_id), float(s["iforest_risk_score"]), float(s.get("shadow_model_proba") or 0.0)))

    out = pd.DataFrame(risks, columns=["TransactionID", "IForestRisk", "ShadowProba"])
    out = out.merge(sample[["TransactionID", "Amount", "CustomerID", "MerchantID", "FraudIndicator"]], on="TransactionID", how="left")
    out = out.sort_values("IForestRisk", ascending=False).head(limit)

    return {"items": out.to_dict(orient="records")}


@app.post("/fraud/score")
def fraud_score(req: ScoreRequest):
    models = try_load_models()
    row = get_tx_row(req.transaction_id)
    return score_transaction(models, row)


@app.post("/fraud/explain")
def fraud_explain(req: ExplainRequest):
    models = try_load_models()
    row = get_tx_row(req.transaction_id)
    return explain_transaction(models, row, top_k=req.top_k)


@app.get("/aml/summary")
def aml_summary():
    df = get_data()
    return graph_summary(df)


@app.get("/aml/smurfing")
def aml_smurfing(mode: str = "fanout", k: int = 10):
    if mode not in ["fanout", "fanin"]:
        raise HTTPException(status_code=400, detail="mode must be fanout or fanin")
    df = get_data()
    return {"mode": mode, "items": detect_smurfing(df, mode=mode, k=k)}


@app.get("/aml/cycles")
def aml_cycles(max_len: int = 6, max_cycles: int = 20):
    df = get_data()
    cycles = detect_cycles(df, max_len=max_len, max_cycles=max_cycles)
    return {"max_len": max_len, "items": cycles}


@app.post("/cases/create")
def cases_create(req: CaseRequest):
    models = try_load_models()
    row = get_tx_row(req.transaction_id)
    tx = row.iloc[0].to_dict()
    if "Timestamp" in tx and isinstance(tx["Timestamp"], pd.Timestamp):
        tx["Timestamp"] = tx["Timestamp"].isoformat()

    score = score_transaction(models, row)
    explain = explain_transaction(models, row, top_k=10)

    # AML quick flags
    df = get_data()
    sm_fanout = detect_smurfing(df, mode="fanout", k=10)
    sm_fanin = detect_smurfing(df, mode="fanin", k=10)

    aml_flags = []
    cid = int(tx.get("CustomerID"))
    mid = int(tx.get("MerchantID"))
    for item in sm_fanout:
        if item.get("customer_id") == cid:
            aml_flags.append(item)
    for item in sm_fanin:
        if item.get("merchant_id") == mid:
            aml_flags.append(item)

    case = {
        "case_id": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "summary": f"TX {req.transaction_id} fraud triage case",
        "analyst": req.analyst,
        "notes": req.notes,
        "transaction": tx,
        "model_outputs": score,
        "xai": explain,
        "aml_flags": aml_flags,
        "evidence": {
            "dataset_files": [
                "Transaction Data/transaction_records.csv",
                "Transaction Data/transaction_metadata.csv",
                "Transaction Amounts/anomaly_scores.csv",
                "Fraudulent Patterns/fraud_indicators.csv",
                "Merchant Information/transaction_category_labels.csv",
                "Customer Profiles/customer_data.csv",
                "Customer Profiles/account_activity.csv",
                "Merchant Information/merchant_data.csv",
                "Fraudulent Patterns/suspicious_activity.csv",
            ],
            "join_key": "TransactionID / CustomerID / MerchantID",
            "evidence_note": "This case is evidence-bundled from joined CSV sources; verify by TransactionID and related IDs.",
        },
        "decision_policy": {
            "principle": "Human-in-the-loop",
            "rule": "If evidence is incomplete or explanation is unstable → escalate (Needs Review).",
        },
    }

    cid = save_case(case, DB_DIR)
    case["case_id"] = cid
    return case


@app.get("/cases")
def cases_list():
    return {"items": list_cases(DB_DIR)}


@app.get("/cases/{case_id}")
def cases_get(case_id: str):
    try:
        return read_case(case_id, DB_DIR)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="case not found")
