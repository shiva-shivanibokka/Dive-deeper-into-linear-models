"use client";
import { useEffect, useState } from "react";

type Schema = { features: string[]; example: Record<string, number>; target: string };
type Prediction = { prediction: number; usd: number; unit: string };

export default function ServingTab() {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [vals, setVals] = useState<Record<string, number>>({});
  const [result, setResult] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/predict")
      .then((r) => r.json())
      .then((s: Schema) => { setSchema(s); setVals(s.example); })
      .catch(() => setErr("could not load model schema"));
  }, []);

  async function run() {
    setBusy(true); setErr(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: vals }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch {
      setErr("prediction request failed");
    } finally {
      setBusy(false);
    }
  }

  if (!schema) return <p className="note">{err ?? "loading model…"}</p>;

  return (
    <div className="demo">
      <p className="callout">
        Unlike the other tabs (which redraw precomputed data), this one makes a <strong>real network
        request</strong> to a serverless function at <code>/api/predict</code>. A Ridge model trained
        offline in Python is served as coefficients; the function standardizes your inputs and computes
        the prediction live. Edit the features and hit predict.
      </p>

      <div className="control-row">
        {schema.features.map((name) => (
          <div className="field" key={name} style={{ flex: "1 1 150px" }}>
            <label htmlFor={name}><span className="lname">{name}</span></label>
            <input
              id={name}
              type="number"
              step="any"
              value={vals[name] ?? 0}
              onChange={(e) => setVals({ ...vals, [name]: parseFloat(e.target.value) })}
              style={{ background: "var(--panel-2)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 10, padding: ".5rem .55rem", fontSize: ".85rem" }}
            />
          </div>
        ))}
      </div>

      <div>
        <button className="tab" style={{ cursor: "pointer" }} onClick={run} disabled={busy} aria-busy={busy}>
          {busy ? "predicting…" : "POST /api/predict"}
        </button>
      </div>

      {err && <p className="note" style={{ color: "var(--bad)" }}>{err}</p>}

      {result && (
        <div className="readout" style={{ textAlign: "center", padding: "1.4rem", borderRadius: 16, border: "1px solid var(--border)", background: "var(--panel-2)" }}>
          <div className="k" style={{ color: "var(--muted)", fontSize: ".72rem", textTransform: "uppercase", letterSpacing: ".08em" }}>predicted {schema.target}</div>
          <div style={{ fontFamily: "var(--font-display)", fontSize: "clamp(2.2rem,6vw,3.2rem)", fontWeight: 700 }}>
            ${result.usd.toLocaleString()}
          </div>
          <div className="note">raw model output: {result.prediction}</div>
        </div>
      )}
    </div>
  );
}
