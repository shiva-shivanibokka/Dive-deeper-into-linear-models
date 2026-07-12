"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Scatter, Dot } from "../lib/svg";

type Deg = { degree: number; fit: [number, number][]; trainErr: number; testErr: number };
type Art = { scatter: [number, number][]; degrees: Deg[] };

export default function BiasVarianceTab() {
  const d = useArtifact<Art>("biasvariance");
  const [i, setI] = useState(0);
  if (!d) return <p className="note">loading…</p>;

  const g = d.degrees[Math.min(i, d.degrees.length - 1)];

  // Chart 1 scales: scatter bounds, extended if the fit runs past them.
  const xs = d.scatter.map((p) => p[0]);
  const ys = d.scatter.map((p) => p[1]);
  const fitYs = g.fit.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys, ...fitYs), y1 = Math.max(...ys, ...fitYs);
  const s1 = makeScales(x0, x1, y0, y1);

  // Chart 2: train/test error across all degrees.
  const trainPts: [number, number][] = d.degrees.map((e) => [e.degree, e.trainErr]);
  const testPts: [number, number][] = d.degrees.map((e) => [e.degree, e.testErr]);
  const maxDeg = Math.max(...d.degrees.map((e) => e.degree));
  const maxErr = Math.max(...d.degrees.map((e) => Math.max(e.trainErr, e.testErr)));
  const s2 = makeScales(1, maxDeg, 0, maxErr);

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field" style={{ maxWidth: 320 }}>
          <label htmlFor="deg"><span className="lname">Polynomial degree: <b>{g.degree}</b></span></label>
          <input id="deg" type="range" min={0} max={d.degrees.length - 1} value={i}
            onChange={(e) => setI(Number(e.target.value))} />
        </div>
        <div className="tiles" style={{ flex: 1 }}>
          <div className="tile"><div className="v">{g.trainErr.toFixed(3)}</div><div className="k">train RMSE</div></div>
          <div className="tile"><div className="v">{g.testErr.toFixed(3)}</div><div className="k">test RMSE</div></div>
        </div>
      </div>
      <div className="grid-2">
        <Chart title="Polynomial fit">
          <Axes x0={x0} x1={x1} y0={y0} y1={y1} sx={s1.sx} sy={s1.sy} xlabel="x" ylabel="y" />
          <Scatter pts={d.scatter} sx={s1.sx} sy={s1.sy} />
          <Line pts={g.fit} sx={s1.sx} sy={s1.sy} stroke="var(--cyan)" width={2.5} />
        </Chart>
        <Chart title="Train vs test error">
          <Axes x0={1} x1={maxDeg} y0={0} y1={maxErr} sx={s2.sx} sy={s2.sy} xlabel="degree" ylabel="RMSE" />
          <Line pts={trainPts} sx={s2.sx} sy={s2.sy} stroke="var(--cyan)" width={2} />
          <Line pts={testPts} sx={s2.sx} sy={s2.sy} stroke="var(--pink)" width={2} />
          <Dot x={g.degree} y={g.trainErr} sx={s2.sx} sy={s2.sy} color="var(--cyan)" />
          <Dot x={g.degree} y={g.testErr} sx={s2.sx} sy={s2.sy} color="var(--pink)" />
        </Chart>
      </div>
      <div className="legend">
        <span className="item"><span className="swatch" style={{ background: "var(--cyan)" }} />train</span>
        <span className="item"><span className="swatch" style={{ background: "var(--pink)" }} />test</span>
      </div>
      <p className="callout">
        A <strong>low degree</strong> underfits — too rigid to follow the signal, so both errors stay high (high bias).
        As the degree grows, <strong>train error keeps dropping</strong> while <strong>test error turns back up</strong>:
        the model is memorizing noise (overfitting, high variance). The sweet spot is the degree that minimizes test error.
      </p>
    </div>
  );
}
