"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Scatter, LegendItem } from "../lib/svg";

type Frame = {
  frac: number;
  outliers: [number, number][];
  ols: [number, number][];
  huber: [number, number][];
  ransac: [number, number][];
  theilsen: [number, number][];
};
type Art = { scatter: [number, number][]; contamination: Frame[] };

export default function RobustTab() {
  const d = useArtifact<Art>("robust");
  const [i, setI] = useState(0);
  if (!d) return <p className="note">loading…</p>;

  const f = d.contamination[Math.min(i, d.contamination.length - 1)];

  const xs = [...d.scatter, ...f.outliers].map((p) => p[0]);
  const ys = [...d.scatter, ...f.outliers].map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const { sx, sy } = makeScales(x0, x1, y0, y1);

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field" style={{ maxWidth: 320 }}>
          <label htmlFor="frac"><span className="lname">Outlier fraction: <b>{(f.frac * 100).toFixed(0)}%</b></span></label>
          <input id="frac" type="range" min={0} max={d.contamination.length - 1} value={Math.min(i, d.contamination.length - 1)}
            onChange={(e) => setI(Number(e.target.value))} />
        </div>
      </div>

      <Chart title="Robust vs ordinary least squares" caption="Points are the data (gray) plus injected outliers (red dots); each colored line is a different fitted regression. Watch how the red OLS line tilts toward the outliers while the robust fits stay near the clean trend.">
        <Axes x0={x0} x1={x1} y0={y0} y1={y1} sx={sx} sy={sy} xlabel="x" ylabel="y" />
        <Scatter pts={d.scatter} sx={sx} sy={sy} color="var(--muted)" />
        <Scatter pts={f.outliers} sx={sx} sy={sy} color="var(--bad)" r={3.5} opacity={1} />
        <Line pts={f.ols} sx={sx} sy={sy} stroke="var(--bad)" width={3} />
        <Line pts={f.huber} sx={sx} sy={sy} stroke="var(--cyan)" width={2} />
        <Line pts={f.ransac} sx={sx} sy={sy} stroke="var(--lime)" width={2} />
        <Line pts={f.theilsen} sx={sx} sy={sy} stroke="var(--violet)" width={2} />
      </Chart>

      <div className="legend">
        <LegendItem color="var(--bad)" label="OLS" term="OLS" />
        <LegendItem color="var(--cyan)" label="Huber" term="Huber" />
        <LegendItem color="var(--lime)" label="RANSAC" term="RANSAC" />
        <LegendItem color="var(--violet)" label="Theil-Sen" term="Theil-Sen" />
        <LegendItem color="var(--bad)" dot label="Outliers" term="Outliers" />
      </div>

      <p className="callout">
        As contamination rises, <strong>OLS</strong> is dragged toward the outliers — a few bad points swing the
        whole line. The <strong>robust fits</strong> (Huber, RANSAC, Theil-Sen) barely move because they down-weight
        or ignore the outliers.
      </p>
    </div>
  );
}
