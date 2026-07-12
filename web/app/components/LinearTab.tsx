"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Scatter } from "../lib/svg";

type PF = { coef: number; r2: number; rmse: number; points: [number, number][]; fit: [number, number][] };
type Art = { features: string[]; perFeature: Record<string, PF> };

export default function LinearTab() {
  const d = useArtifact<Art>("linear");
  const [feat, setFeat] = useState<string | null>(null);
  if (!d) return <p className="note">loading…</p>;

  const name = feat ?? d.features[0];
  const pf = d.perFeature[name];
  const xs = pf.points.map((p) => p[0]);
  const ys = pf.points.map((p) => p[1]);
  const { sx, sy } = makeScales(Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys));

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field" style={{ maxWidth: 260 }}>
          <label htmlFor="feat"><span className="lname">Feature</span></label>
          <select id="feat" value={name} onChange={(e) => setFeat(e.target.value)}>
            {d.features.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="tiles" style={{ flex: 1 }}>
          <div className="tile"><div className="v">{pf.r2.toFixed(3)}</div><div className="k">R²</div></div>
          <div className="tile"><div className="v">{pf.rmse.toFixed(2)}</div><div className="k">RMSE</div></div>
          <div className="tile"><div className="v">{pf.coef.toFixed(3)}</div><div className="k">Slope</div></div>
        </div>
      </div>

      <Chart title={`Least-squares fit on ${name}`}>
        <Axes x0={Math.min(...xs)} x1={Math.max(...xs)} y0={Math.min(...ys)} y1={Math.max(...ys)}
          sx={sx} sy={sy} xlabel={name} ylabel="median value ($100k)" />
        <Scatter pts={pf.points} sx={sx} sy={sy} />
        <Line pts={pf.fit} sx={sx} sy={sy} stroke="var(--pink)" width={2.5} />
      </Chart>

      <p className="callout">
        Each feature is fit on its own here. A steep slope with a high R² (like <strong>MedInc</strong>) means that
        feature alone explains a lot of the variation in house value; a flat, low-R² feature explains little.
      </p>
    </div>
  );
}
