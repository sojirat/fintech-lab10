export const metadata = {
  title: "FinTech Lab 10 — Fraud & AML Analyst Console",
  description: "Lecture 10 lab UI (dataset-backed)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial", margin: 0, background: "#0b1220", color: "#e6edf3" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, marginBottom: 18 }}>
            <div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>FinTech Lab 10</div>
              <div style={{ opacity: 0.8, marginTop: 4 }}>Fraud Detection + AML Graph Hunt + XAI + Case Pack</div>
            </div>
            <div style={{ fontSize: 12, opacity: 0.7 }}>UI: Next.js • API: FastAPI • Dataset: CSV bundle</div>
          </div>
          {children}
          <div style={{ opacity: 0.6, fontSize: 12, marginTop: 30 }}>
            Classroom scaffold • Evidence-bound decisions • Human-in-the-loop
          </div>
        </div>
      </body>
    </html>
  );
}
