"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";

type AnyObj = Record<string, any>;

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "#111a2e", border: "1px solid #1f2a44", borderRadius: 14, padding: 16, marginBottom: 14 }}>
      <div style={{ fontWeight: 700, marginBottom: 10 }}>{title}</div>
      {children}
    </div>
  );
}

function Button({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "#2563eb",
        color: "white",
        border: "none",
        borderRadius: 10,
        padding: "10px 12px",
        cursor: "pointer",
        fontWeight: 600,
      }}
    >
      {label}
    </button>
  );
}

export default function Page() {
  const [summary, setSummary] = useState<AnyObj | null>(null);
  const [trainRes, setTrainRes] = useState<AnyObj | null>(null);
  const [top, setTop] = useState<AnyObj[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [tx, setTx] = useState<AnyObj | null>(null);
  const [score, setScore] = useState<AnyObj | null>(null);
  const [explain, setExplain] = useState<AnyObj | null>(null);
  const [amlSummary, setAmlSummary] = useState<AnyObj | null>(null);
  const [smurfFanout, setSmurfFanout] = useState<AnyObj | null>(null);
  const [smurfFanin, setSmurfFanin] = useState<AnyObj | null>(null);
  const [cycles, setCycles] = useState<AnyObj | null>(null);
  const [casePack, setCasePack] = useState<AnyObj | null>(null);
  const [cases, setCases] = useState<AnyObj | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function safe<T>(fn: () => Promise<T>) {
    setErr(null);
    try {
      return await fn();
    } catch (e: any) {
      setErr(e?.message || String(e));
      throw e;
    }
  }

  const refreshSummary = () => safe(async () => setSummary(await api.summary()));
  const train = () => safe(async () => setTrainRes(await api.train()));
  const loadTop = () => safe(async () => setTop((await api.top(30)).items || []));
  const loadAml = () => safe(async () => setAmlSummary(await api.amlSummary()));
  const runSmurf = () => safe(async () => {
    setSmurfFanout(await api.smurfing("fanout", 10));
    setSmurfFanin(await api.smurfing("fanin", 10));
  });
  const runCycles = () => safe(async () => setCycles(await api.cycles(6, 20)));
  const loadCases = () => safe(async () => setCases(await api.listCases()));

  async function selectTx(id: number) {
    setSelected(id);
    setCasePack(null);
    await safe(async () => {
      const t = await api.tx(id);
      setTx(t);
      setScore(await api.score(id));
      setExplain(await api.explain(id, 10));
    });
  }

  async function createCase() {
    if (!selected) return;
    await safe(async () => setCasePack(await api.createCase(selected)));
    await loadCases();
  }

  useEffect(() => {
    refreshSummary();
    loadAml();
  }, []);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.15fr 0.85fr", gap: 14 }}>
      <div>
        <Card title="1) Dataset Status">
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
            <Button label="Refresh Summary" onClick={refreshSummary} />
            <Button label="Train Models" onClick={train} />
            <Button label="Load Top Anomalies" onClick={loadTop} />
          </div>
          {summary ? (
            <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(summary, null, 2)}</pre>
          ) : (
            <div style={{ opacity: 0.7 }}>Loading…</div>
          )}
          {trainRes && (
            <>
              <div style={{ marginTop: 10, fontWeight: 700 }}>Training Result</div>
              <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(trainRes, null, 2)}</pre>
            </>
          )}
        </Card>

        <Card title="2) Top Anomalies (click a TransactionID)">
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left", opacity: 0.85 }}>
                  {["TransactionID", "Amount", "IForestRisk", "ShadowProba", "FraudIndicator"].map((h) => (
                    <th key={h} style={{ padding: "8px 6px", borderBottom: "1px solid #1f2a44" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {top.map((r) => (
                  <tr key={r.TransactionID} style={{ cursor: "pointer", background: selected === r.TransactionID ? "#0f2a60" : "transparent" }} onClick={() => selectTx(Number(r.TransactionID))}>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #1f2a44" }}>{r.TransactionID}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #1f2a44" }}>{Number(r.Amount).toFixed(2)}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #1f2a44" }}>{Number(r.IForestRisk).toFixed(4)}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #1f2a44" }}>{Number(r.ShadowProba).toFixed(4)}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #1f2a44" }}>{r.FraudIndicator}</td>
                  </tr>
                ))}
                {top.length === 0 && (
                  <tr><td colSpan={5} style={{ padding: 10, opacity: 0.7 }}>No items. Click “Train Models” then “Load Top Anomalies”.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="3) Selected Transaction — Score + Explain + Case Pack">
          {!selected && <div style={{ opacity: 0.7 }}>Select a transaction from Top Anomalies.</div>}
          {selected && (
            <>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
                <Button label="Create Case Pack" onClick={createCase} />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>Transaction</div>
                  <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(tx, null, 2)}</pre>
                </div>
                <div>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>Model Outputs</div>
                  <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(score, null, 2)}</pre>
                  <div style={{ fontWeight: 700, marginTop: 10, marginBottom: 6 }}>Explain (Top Factors)</div>
                  <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(explain, null, 2)}</pre>
                </div>
              </div>
              {casePack && (
                <>
                  <div style={{ fontWeight: 700, marginTop: 10, marginBottom: 6 }}>Case Pack</div>
                  <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(casePack, null, 2)}</pre>
                </>
              )}
            </>
          )}
        </Card>
      </div>

      <div>
        <Card title="AML Graph Hunt">
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
            <Button label="Graph Summary" onClick={loadAml} />
            <Button label="Smurfing (Fan-out/Fan-in)" onClick={runSmurf} />
            <Button label="Find Cycles" onClick={runCycles} />
          </div>
          {amlSummary && <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(amlSummary, null, 2)}</pre>}
          {smurfFanout && (
            <>
              <div style={{ fontWeight: 700, marginTop: 10 }}>Smurfing — Fan-out</div>
              <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(smurfFanout, null, 2)}</pre>
            </>
          )}
          {smurfFanin && (
            <>
              <div style={{ fontWeight: 700, marginTop: 10 }}>Smurfing — Fan-in</div>
              <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(smurfFanin, null, 2)}</pre>
            </>
          )}
          {cycles && (
            <>
              <div style={{ fontWeight: 700, marginTop: 10 }}>Cycles / Rings</div>
              <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(cycles, null, 2)}</pre>
            </>
          )}
        </Card>

        <Card title="Cases (Human-in-the-loop)">
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
            <Button label="Refresh Case List" onClick={loadCases} />
          </div>
          {cases ? (
            <pre style={{ whiteSpace: "pre-wrap", margin: 0, opacity: 0.9 }}>{JSON.stringify(cases, null, 2)}</pre>
          ) : (
            <div style={{ opacity: 0.7 }}>Click “Refresh Case List”.</div>
          )}
        </Card>

        {err && (
          <div style={{ background: "#3b1d1d", border: "1px solid #7f1d1d", padding: 12, borderRadius: 12 }}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Error</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{err}</div>
          </div>
        )}
      </div>
    </div>
  );
}
