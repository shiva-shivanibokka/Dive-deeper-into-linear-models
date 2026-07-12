"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Scatter, Band } from "../lib/svg";

type Prior = { lambda: number; mean: [number, number][]; band: [number, number, number][] };
type Art = { scatter: [number, number][]; priors: Prior[] };

export default function BayesianTab() {
  const d = useArtifact<Art>("bayesian");
  const [i, setI] = useState(0);
  if (!d) return <p className="note">loading…</p>;

  const p = d.priors[Math.min(i, d.priors.length - 1)];
  const xs = d.scatter.map((s) => s[0]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ys = d.scatter.map((s) => s[1]);
  const y0 = Math.min(...ys, ...p.band.map((b) => b[1]));
  const y1 = Math.max(...ys, ...p.band.map((b) => b[2]));
  const { sx, sy } = makeScales(x0, x1, y0, y1);

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field" style={{ maxWidth: 320 }}>
          <label htmlFor="lam"><span className="lname">Prior strength λ: <b>{p.lambda}</b></span></label>
          <input id="lam" type="range" min={0} max={d.priors.length - 1} value={i}
            onChange={(e) => setI(Number(e.target.value))} />
        </div>
      </div>
      <Chart title={`posterior mean at λ=${p.lambda}`}>
        <Axes x0={x0} x1={x1} y0={y0} y1={y1} sx={sx} sy={sy} xlabel="x" ylabel="y" />
        <Band pts={p.band} sx={sx} sy={sy} fill="var(--violet)" />
        <Scatter pts={d.scatter} sx={sx} sy={sy} />
        <Line pts={p.mean} sx={sx} sy={sy} stroke="var(--violet)" width={2.5} />
      </Chart>
      <p className="callout">
        A Bayesian prior regularizes the fit. A <strong>larger λ</strong> pulls the posterior mean
        <strong> toward zero</strong> and <strong>widens the uncertainty band</strong>, especially
        away from the data where the observations no longer pin the curve down.
      </p>
    </div>
  );
}
