from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx


def build_graph(tx_df: pd.DataFrame, bidirectional: bool = True) -> nx.DiGraph:
    """Build a Customer↔Merchant directed graph.
    sender = CustomerID, receiver = MerchantID.
    Optionally add reverse edges to enable ring/cycle signals.
    """
    G = nx.DiGraph()

    for _, r in tx_df.iterrows():
        c = f"C:{int(r['CustomerID'])}"
        m = f"M:{int(r['MerchantID'])}"
        amt = float(r.get("Amount", 0.0) or 0.0)
        tid = int(r.get("TransactionID"))
        # add nodes
        G.add_node(c, node_type="customer")
        G.add_node(m, node_type="merchant")
        # add edge with attributes
        G.add_edge(c, m, amount=amt, tx_id=tid)
        if bidirectional:
            G.add_edge(m, c, amount=amt, tx_id=tid)

    return G


def detect_smurfing(tx_df: pd.DataFrame, mode: str = "fanout", k: int = 10, small_q: float = 0.25) -> List[Dict[str, Any]]:
    """Heuristic smurfing-like detection on customer↔merchant edges.
    - fanout: one customer -> many merchants with small amounts
    - fanin: many customers -> one merchant with small amounts
    """
    d = tx_df.copy()
    if "Amount" not in d.columns:
        return []

    thr = float(d["Amount"].quantile(small_q))
    small = d[d["Amount"] <= thr].copy()

    results: List[Dict[str, Any]] = []

    if mode == "fanout":
        g = small.groupby("CustomerID")["MerchantID"].nunique().reset_index(name="distinct_merchants")
        hit = g[g["distinct_merchants"] >= k].sort_values("distinct_merchants", ascending=False)
        for _, r in hit.head(50).iterrows():
            cid = int(r["CustomerID"])
            sub = small[small["CustomerID"] == cid]
            results.append({
                "pattern": "smurfing_fanout",
                "customer_id": cid,
                "distinct_merchants": int(r["distinct_merchants"]),
                "small_amount_threshold": thr,
                "example_tx_ids": sub["TransactionID"].head(10).astype(int).tolist(),
            })
    else:
        g = small.groupby("MerchantID")["CustomerID"].nunique().reset_index(name="distinct_customers")
        hit = g[g["distinct_customers"] >= k].sort_values("distinct_customers", ascending=False)
        for _, r in hit.head(50).iterrows():
            mid = int(r["MerchantID"])
            sub = small[small["MerchantID"] == mid]
            results.append({
                "pattern": "smurfing_fanin",
                "merchant_id": mid,
                "distinct_customers": int(r["distinct_customers"]),
                "small_amount_threshold": thr,
                "example_tx_ids": sub["TransactionID"].head(10).astype(int).tolist(),
            })

    return results


def detect_cycles(tx_df: pd.DataFrame, max_len: int = 6, max_cycles: int = 20) -> List[Dict[str, Any]]:
    """Detect short cycles/rings.
    Since the dataset is customer↔merchant, we:
    - build a bidirectional graph (enables alternating C-M-C-M cycles)
    - run a bounded cycle search on a reduced subgraph for performance
    """
    G = build_graph(tx_df, bidirectional=True)

    # Reduce to top-degree nodes to keep cycle search fast (classroom-friendly)
    deg = dict(G.degree())
    top_nodes = sorted(deg, key=deg.get, reverse=True)[:400]
    H = G.subgraph(top_nodes).copy()

    cycles_out: List[Dict[str, Any]] = []
    try:
        for cyc in nx.simple_cycles(H):
            if 3 <= len(cyc) <= max_len:
                cycles_out.append({"cycle": cyc, "length": len(cyc)})
            if len(cycles_out) >= max_cycles:
                break
    except Exception:
        return []

    return cycles_out


def graph_summary(tx_df: pd.DataFrame) -> Dict[str, Any]:
    G = build_graph(tx_df, bidirectional=True)
    n_customers = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "customer")
    n_merchants = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "merchant")

    deg = dict(G.degree())
    top = sorted(deg.items(), key=lambda t: t[1], reverse=True)[:10]

    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "customers": int(n_customers),
        "merchants": int(n_merchants),
        "top_degree_nodes": [{"node": n, "degree": int(v)} for n, v in top],
    }
