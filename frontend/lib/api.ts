const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function getJSON(path: string) {
  const r = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function postJSON(path: string, body: any) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export const api = {
  healthz: () => getJSON("/healthz"),
  summary: () => getJSON("/dataset/summary"),
  reload: () => postJSON("/dataset/reload", {}),
  train: () => postJSON("/fraud/train", {}),
  top: (limit = 20) => getJSON(`/fraud/top?limit=${limit}`),
  tx: (id: number) => getJSON(`/transactions/${id}`),
  score: (id: number) => postJSON("/fraud/score", { transaction_id: id }),
  explain: (id: number, top_k = 10) => postJSON("/fraud/explain", { transaction_id: id, top_k }),
  amlSummary: () => getJSON("/aml/summary"),
  smurfing: (mode = "fanout", k = 10) => getJSON(`/aml/smurfing?mode=${mode}&k=${k}`),
  cycles: (max_len = 6, max_cycles = 20) => getJSON(`/aml/cycles?max_len=${max_len}&max_cycles=${max_cycles}`),
  createCase: (id: number) => postJSON("/cases/create", { transaction_id: id }),
  listCases: () => getJSON("/cases"),
  getCase: (case_id: string) => getJSON(`/cases/${case_id}`),
};
