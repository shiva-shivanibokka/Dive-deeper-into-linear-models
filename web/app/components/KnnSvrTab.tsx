"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Scatter, Band } from "../lib/svg";

type Knn = { k: number; fit: [number, number][] };
type Svr = { eps: number; C: number; fit: [number, number][]; tube: [number, number, number][] };
type Art = { scatter: [number, number][]; knn: Knn[]; svr: Svr[] };

export default function KnnSvrTab() {
  const d = useArtifact<Art>("knnsvr");
  const [mode, setMode] = useState<"KNN" | "SVR">("KNN");
  const [ki, setKi] = useState(0);
  const [si, setSi] = useState(0);
  if (!d) return <p className="note">loading…</p>;

  const knn = d.knn[Math.min(ki, d.knn.length - 1)];
  const svr = d.svr[Math.min(si, d.svr.length - 1)];
  const fit = mode === "KNN" ? knn.fit : svr.fit;

  const xs = d.scatter.map((s) => s[0]);
  const ys = d.scatter.map((s) => s[1]);
  const { sx, sy } = makeScales(Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys));

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field">
          <label><span className="lname">Model</span></label>
          <div className="seg">
            <button aria-pressed={mode === "KNN"} onClick={() => setMode("KNN")}>KNN</button>
            <button aria-pressed={mode === "SVR"} onClick={() => setMode("SVR")}>SVR</button>
          </div>
        </div>
        {mode === "KNN" ? (
          <div className="field">
            <label><span className="lname">Neighbors k</span></label>
            <div className="seg">
              {d.knn.map((e, idx) => (
                <button key={e.k} aria-pressed={idx === ki} onClick={() => setKi(idx)}>k={e.k}</button>
              ))}
            </div>
          </div>
        ) : (
          <div className="field">
            <label><span className="lname">Hyperparameters</span></label>
            <div className="seg">
              {d.svr.map((e, idx) => (
                <button key={`${e.eps}-${e.C}`} aria-pressed={idx === si} onClick={() => setSi(idx)}>
                  ε={e.eps} C={e.C}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      <Chart title={mode === "KNN" ? `KNN fit (k=${knn.k})` : `SVR fit (ε=${svr.eps}, C=${svr.C})`} caption="x and y are the data; grey dots are the samples and the coloured line is the model's prediction. In SVR mode the shaded band is the ε-tube — points inside it cost nothing, so only points on or outside it shape the fit.">
        <Axes x0={Math.min(...xs)} x1={Math.max(...xs)} y0={Math.min(...ys)} y1={Math.max(...ys)}
          sx={sx} sy={sy} xlabel="x" ylabel="y" />
        {mode === "SVR" && <Band pts={svr.tube} sx={sx} sy={sy} fill="var(--pink)" />}
        <Scatter pts={d.scatter} sx={sx} sy={sy} />
        <Line pts={fit} sx={sx} sy={sy} stroke={mode === "KNN" ? "var(--cyan)" : "var(--pink)"} width={2.5} />
      </Chart>
      <p className="callout">
        <strong>KNN</strong> predicts by averaging the nearest points — a <strong>small k</strong> is jagged
        and noise-sensitive, a <strong>large k</strong> is smooth but blunts real structure.
        <strong> SVR</strong> fits an <strong>ε-tube</strong> around the curve and pays no penalty for
        points inside it, so only the boundary points (support vectors) shape the fit; larger C punishes
        outside-tube errors harder.
      </p>
    </div>
  );
}
