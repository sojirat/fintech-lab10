import os
from pathlib import Path

# Point the API to the packaged dataset inside this repo
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["DATA_DIR"] = str(REPO_ROOT.parent / "data")
os.environ["MODELS_DIR"] = str(REPO_ROOT / "models")
os.environ["DB_DIR"] = str(REPO_ROOT / "db")
os.environ["REDIS_URL"] = "redis://localhost:6379/0"  # not required for tests

import sys
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"

def test_summary_and_train_and_score():
    r = client.get("/dataset/summary")
    assert r.status_code == 200
    j = r.json()
    assert j["rows"] > 0

    r = client.post("/fraud/train")
    assert r.status_code == 200
    t = r.json()
    assert t["status"] == "trained"

    r = client.get("/fraud/top?limit=5")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) <= 5
    tx_id = int(items[0]["TransactionID"])

    r = client.post("/fraud/score", json={"transaction_id": tx_id})
    assert r.status_code == 200
    s = r.json()
    assert "iforest_risk_score" in s

    r = client.post("/fraud/explain", json={"transaction_id": tx_id, "top_k": 5})
    assert r.status_code == 200
    e = r.json()
    assert "top_factors" in e
