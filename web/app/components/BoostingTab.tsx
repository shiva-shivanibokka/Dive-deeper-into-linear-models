"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line } from "../lib/svg";

type Art = {
  features: string[];
  gbmImportance: number[];
  xgbImportance: number[];
  pdp: { feature: string; x: number[]; y: number[] };
};

export default function BoostingTab() {
  const d = useArtifact<Art>("boosting");
  const [model, setModel] = useState<"gbm" | "xgb">("gbm");
  if (!d) return <p className="note">loading…</p>;

  const imp = model === "gbm" ? d.gbmImportance : d.xgbImportance;
  const rows = d.features
    .map((name, i) => ({ name, v: imp[i] }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 12);
  const maxV = Math.max(...rows.map((r) => r.v)) || 1;

  const pdpPts = d.pdp.x.map((x, i) => [x, d.pdp.y[i]] as [number, number]);
  const xs = d.pdp.x, ys = d.pdp.y;
  const { sx, sy } = makeScales(Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys));

  return (
    <div className="demo">
      <div className="control-row">
        <div className="seg" role="group" aria-label="Model">
          <button aria-pressed={model === "gbm"} onClick={() => setModel("gbm")}>Gradient Boosting</button>
          <button aria-pressed={model === "xgb"} onClick={() => setModel("xgb")}>XGBoost</button>
        </div>
      </div>

      <div>
        <p className="section-label">Top feature importances — {model === "gbm" ? "Gradient Boosting" : "XGBoost"}</p>
        <div className="bars">
          {rows.map((r) => (
            <div className="bar-row" key={r.name}>
              <span className="name" title={r.name}>{r.name}</span>
              <div className="bar-track"><div className="fill" style={{ width: `${(r.v / maxV) * 100}%` }} /></div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <p className="section-label">Partial dependence — {d.pdp.feature}</p>
        <Chart title={`Partial dependence of ${d.pdp.feature}`}>
          <Axes x0={Math.min(...xs)} x1={Math.max(...xs)} y0={Math.min(...ys)} y1={Math.max(...ys)}
            sx={sx} sy={sy} xlabel={d.pdp.feature} ylabel="avg. prediction" />
          <Line pts={pdpPts} sx={sx} sy={sy} stroke="var(--lime)" width={2.5} />
        </Chart>
      </div>

      <p className="callout">
        Boosting builds many small trees, each correcting the last one's errors. <strong>Feature importance</strong>{" "}
        ranks how much each measurement was used to split; <strong>partial dependence</strong> traces the average
        effect of the single most important feature as it varies.
      </p>
    </div>
  );
}
