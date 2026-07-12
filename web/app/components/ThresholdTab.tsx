"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line, Dot } from "../lib/svg";

type Row = {
  t: number; tp: number; fp: number; tn: number; fn: number;
  precision: number; recall: number; f1: number;
};
type Art = {
  scores: number[];
  labels: number[];
  roc: [number, number][];
  pr: [number, number][];
  thresholds: Row[];
};

export default function ThresholdTab() {
  const d = useArtifact<Art>("threshold");
  const [i, setI] = useState(0);
  if (!d) return <p className="note">loading…</p>;

  const k = Math.min(i, d.thresholds.length - 1);
  const r = d.thresholds[k];
  const { tp, fp, tn, fn } = r;

  // ponytail: guard divide-by-zero — empty denominator collapses to 0.
  const fpr = fp + tn === 0 ? 0 : fp / (fp + tn);
  const tpr = tp + fn === 0 ? 0 : tp / (tp + fn);

  const roc = makeScales(0, 1, 0, 1);
  const pr = makeScales(0, 1, 0, 1);

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field" style={{ maxWidth: 320 }}>
          <label htmlFor="thr"><span className="lname">Decision threshold: <b>{r.t.toFixed(2)}</b></span></label>
          <input id="thr" type="range" min={0} max={d.thresholds.length - 1} value={k}
            onChange={(e) => setI(Number(e.target.value))} />
        </div>
        <div className="tiles" style={{ flex: 1 }}>
          <div className="tile"><div className="v">{r.precision.toFixed(3)}</div><div className="k">precision</div></div>
          <div className="tile"><div className="v">{r.recall.toFixed(3)}</div><div className="k">recall</div></div>
          <div className="tile"><div className="v">{r.f1.toFixed(3)}</div><div className="k">F1</div></div>
        </div>
      </div>

      <p className="section-label">Confusion matrix at this threshold</p>
      <div className="cm">
        <div className="cell ok"><div className="n">{tp}</div><div className="t">True Positive</div></div>
        <div className="cell err"><div className="n">{fp}</div><div className="t">False Positive</div></div>
        <div className="cell err"><div className="n">{fn}</div><div className="t">False Negative</div></div>
        <div className="cell ok"><div className="n">{tn}</div><div className="t">True Negative</div></div>
      </div>

      <div className="grid-2">
        <Chart title="ROC curve" caption="true-positive rate vs false-positive rate; the dot is your current threshold — up and to the left is better.">
          <Axes x0={0} x1={1} y0={0} y1={1} sx={roc.sx} sy={roc.sy} xlabel="false positive rate" ylabel="true positive rate" />
          <Line pts={[[0, 0], [1, 1]]} sx={roc.sx} sy={roc.sy} stroke="var(--muted)" width={1.5} dash="3 3" />
          <Line pts={d.roc} sx={roc.sx} sy={roc.sy} stroke="var(--cyan)" width={2.5} />
          <Dot x={fpr} y={tpr} sx={roc.sx} sy={roc.sy} color="var(--lime)" />
        </Chart>
        <Chart title="Precision–Recall" caption="precision vs recall; the dot is your current threshold.">
          <Axes x0={0} x1={1} y0={0} y1={1} sx={pr.sx} sy={pr.sy} xlabel="recall" ylabel="precision" />
          <Line pts={d.pr} sx={pr.sx} sy={pr.sy} stroke="var(--pink)" width={2.5} />
          <Dot x={r.recall} y={r.precision} sx={pr.sx} sy={pr.sy} color="var(--lime)" />
        </Chart>
      </div>

      <p className="callout">
        <strong>Lowering the threshold</strong> catches more positives (higher <strong>recall</strong>) but raises
        false alarms (lower <strong>precision</strong>). As you drag the slider, the dots slide along the ROC and
        precision–recall curves, tracing the trade-off you accept at each operating point.
      </p>
    </div>
  );
}
