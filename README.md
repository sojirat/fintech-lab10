# FinTech Lab – Lecture 10 (Fraud + AML Graph + XAI) — Dataset-Backed Version

This lab is built to match **Lecture 10: AI-Driven Fraud Detection**.
It uses the attached dataset (CSV folders) and implements:

- **Imbalanced data handling** (upsampling)
- **Isolation Forest** (unsupervised fraud/anomaly scoring)
- **Graph-based AML hunt** (Smurfing star + Cycle / ring signals)
- **Explainability (XAI)** with a robust fallback when SHAP is unavailable
- **Human-in-the-loop Case Pack** (evidence bundle)

## 1) Services

| Service | Purpose | Host |
|---|---|---|
| `api` | FastAPI backend (train/score/explain/AML/cases) | http://localhost:8000 |
| `frontend` | Next.js “Analyst Console” UI | http://localhost:3000 |
| `redis` | Rate limiting (lecture real-time architecture concept) | localhost:6379 |

## 2) Quick Start

```bash
docker compose up -d --build
```

Open:
- UI: http://localhost:3000
- API docs: http://localhost:8000/docs

## 3) In-class Demo Flow (recommended)

1) **Dataset summary** (UI → refresh)
2) **Train** (UI → Train Models)
3) **Top anomalies** (UI → Load Top Anomalies)
4) Select a transaction → **Score** → **Explain**
5) Run **AML Smurfing** + **Cycle** hunt
6) **Create Case** → download the JSON evidence bundle

## 4) Dataset (from the attached zip)

The dataset is mounted into the API container at `/app/data`.

- `Transaction Data/transaction_records.csv`
- `Transaction Data/transaction_metadata.csv`
- `Transaction Amounts/anomaly_scores.csv`
- `Fraudulent Patterns/fraud_indicators.csv`
- `Customer Profiles/customer_data.csv`
- `Customer Profiles/account_activity.csv`
- `Merchant Information/merchant_data.csv`
- `Merchant Information/transaction_category_labels.csv`
- `Fraudulent Patterns/suspicious_activity.csv`

## 5) Key API Endpoints

- `GET /healthz`
- `GET /dataset/summary`
- `POST /fraud/train`
- `GET /fraud/top?limit=30`
- `POST /fraud/score`
- `POST /fraud/explain`
- `GET /aml/smurfing?mode=fanout&k=10`
- `GET /aml/cycles?max_len=6&max_cycles=20`
- `POST /cases/create`

## 6) Notes on “Graph Edges” for this dataset

Lecture 10 uses `G.add_edge(sender, receiver)` for transfers.
This dataset is **customer → merchant** transactions, so we model:

- sender = `CustomerID`
- receiver = `MerchantID`

To enable “cycle/ring” signals, we add reverse edges (merchant → customer) and also compute a customer-projection graph (customers linked by shared merchants).

## License / Classroom Use
Classroom-only lab scaffold, not production banking software.


## Lab Workbook & Notebooks

- `docs/Lab10_Workbook.md` — คู่มือแล็บ (Task A–E)
- `notebooks/Lab10_Student.ipynb`
- `notebooks/Lab10_Instructor.ipynb`
