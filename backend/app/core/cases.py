from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def new_case_id() -> str:
    return uuid.uuid4().hex[:12]


def save_case(case: Dict[str, Any], db_dir: str) -> str:
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    cid = case.get("case_id") or new_case_id()
    case["case_id"] = cid
    p = Path(db_dir) / f"case_{cid}.json"
    p.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    return cid


def list_cases(db_dir: str) -> List[Dict[str, Any]]:
    p = Path(db_dir)
    if not p.exists():
        return []
    out = []
    for fp in sorted(p.glob("case_*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            out.append({"case_id": obj.get("case_id"), "created_at": obj.get("created_at"), "summary": obj.get("summary")})
        except Exception:
            continue
    return out


def read_case(case_id: str, db_dir: str) -> Dict[str, Any]:
    fp = Path(db_dir) / f"case_{case_id}.json"
    if not fp.exists():
        raise FileNotFoundError(case_id)
    return json.loads(fp.read_text(encoding="utf-8"))
