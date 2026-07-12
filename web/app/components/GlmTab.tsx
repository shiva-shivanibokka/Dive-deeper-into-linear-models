"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Scatter } from "../lib/svg";

type ModelFit = { name: string; pred: number[]; r2: number; fracNegative: number };
type Art = {
  actual: number[];
  models: ModelFit[];
  tweediePowers: { p: number; pred: number[] }[];
};

export default function GlmTab() {
  const d = useArtifact<Art>("glm");
  const [name, setName] = useState<string | null>(null);
  if (!d) return <p className="note">loading…</p>;

  const sel = d.models.find((m) => m.name === (name ?? d.models[0].name)) ?? d.models[0];

  const allPreds = d.models.flatMap((m) => m.pred);
  const lo = Math.min(...d.actual, ...allPreds);
  const hi = Math.max(...d.actual, ...allPreds);
  const { sx, sy } = makeScales(lo, hi, lo, hi);

  const pts: [number, number][] = d.actual.map((a, idx) => [a, sel.pred[idx]]);

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field">
          <label><span className="lname">Model</span></label>
          <div className="seg">
            {d.models.map((m) => (
              <button key={m.name} aria-pressed={sel.name === m.name} onClick={() => setName(m.name)}>{m.name}</button>
            ))}
          </div>
        </div>
        <div className="tiles" style={{ flex: 1 }}>
          <div className="tile"><div className="v">{sel.r2.toFixed(3)}</div><div className="k">R²</div></div>
          <div className="tile"><div className="v">{(sel.fracNegative * 100).toFixed(1)}%</div><div className="k">Negative preds</div></div>
        </div>
      </div>

      <Chart title="Predicted vs actual" caption="Each dot is one house: x is the actual value, y is the model's prediction. The dashed diagonal is perfect prediction, and the red horizontal line marks zero — dots below it are impossible negative predictions.">
        <Axes x0={lo} x1={hi} y0={lo} y1={hi} sx={sx} sy={sy} xlabel="actual" ylabel="predicted" />
        <Line pts={[[lo, lo], [hi, hi]]} sx={sx} sy={sy} stroke="var(--muted)" width={1.5} dash="4 3" />
        <Line pts={[[lo, 0], [hi, 0]]} sx={sx} sy={sy} stroke="var(--bad)" width={1.5} dash="4 3" />
        <Scatter pts={pts} sx={sx} sy={sy} color="var(--cyan)" />
      </Chart>

      <p className="callout">
        Plain <strong>OLS</strong> can predict impossible <strong>negative house values</strong> (note the nonzero
        negative-prediction share below the red line). <strong>Poisson, Gamma, and Tweedie</strong> GLMs use a link
        function that keeps every prediction positive.
      </p>
    </div>
  );
}
