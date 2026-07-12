"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Scatter, Band } from "../lib/svg";

type Q = { q: number; fit: [number, number][] };
type Art = { scatter: [number, number][]; quantiles: Q[] };

export default function QuantileTab() {
  const d = useArtifact<Art>("quantile");
  const [i, setI] = useState(0);
  if (!d) return <p className="note">loading…</p>;

  const qs = d.quantiles;
  const sel = qs[Math.min(i, qs.length - 1)];
  const lo = qs[0], hi = qs[qs.length - 1];
  const median = qs.reduce((a, b) => (Math.abs(b.q - 0.5) < Math.abs(a.q - 0.5) ? b : a), qs[0]);

  // Interval band between lowest and highest quantile.
  const band: [number, number, number][] = lo.fit.map((pt, k) => [pt[0], pt[1], hi.fit[k][1]]);

  const xs = d.scatter.map((s) => s[0]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ys = d.scatter.map((s) => s[1]);
  const y0 = Math.min(...ys, ...band.map((b) => b[1]));
  const y1 = Math.max(...ys, ...band.map((b) => b[2]));
  const { sx, sy } = makeScales(x0, x1, y0, y1);

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field" style={{ maxWidth: 320 }}>
          <label htmlFor="q"><span className="lname">Quantile: <b>{sel.q}</b></span></label>
          <input id="q" type="range" min={0} max={qs.length - 1} value={i}
            onChange={(e) => setI(Number(e.target.value))} />
        </div>
      </div>
      <Chart title={`quantile fit q=${sel.q}`} caption="x and y are the data; dots are observations. The solid lime line is the selected quantile fit, the dashed grey line is the median, and the shaded band spans the lowest-to-highest quantile — a prediction interval that should contain most points.">
        <Axes x0={x0} x1={x1} y0={y0} y1={y1} sx={sx} sy={sy} xlabel="x" ylabel="y" />
        <Band pts={band} sx={sx} sy={sy} fill="var(--lime)" opacity={0.1} />
        <Scatter pts={d.scatter} sx={sx} sy={sy} />
        <Line pts={median.fit} sx={sx} sy={sy} stroke="var(--muted)" width={1} dash="4 3" />
        <Line pts={sel.fit} sx={sx} sy={sy} stroke="var(--lime)" width={2.5} />
      </Chart>
      <p className="callout">
        Quantile regression predicts a <strong>percentile</strong> of the target, not its mean —
        the {sel.q} line is exceeded by roughly {(sel.q * 100).toFixed(0)}% of points below it.
        Stacking the lowest and highest quantiles yields a <strong>prediction interval</strong> (the shaded band).
      </p>
    </div>
  );
}
