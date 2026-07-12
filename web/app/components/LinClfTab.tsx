"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { makeScales, Chart, Axes, Points, Heat, LegendItem, CLASS_COLORS } from "../lib/svg";

const CLASS_NAMES = ["Iris setosa", "Iris versicolor", "Iris virginica"];

type Grid = { x0: number; x1: number; y0: number; y1: number; step: number };
type LinClf = {
  points: [number, number, number][];
  grid: Grid;
  perceptronIters: { iters: number; grid: number[][] }[];
  ridge: number[][];
};

export default function LinClfTab() {
  const [mode, setMode] = useState<"perceptron" | "ridge">("perceptron");
  const [iterIdx, setIterIdx] = useState(0);
  const d = useArtifact<LinClf>("linclf");
  if (!d) return <p className="note">loading…</p>;

  const g = d.grid;
  const { sx, sy } = makeScales(g.x0, g.x1, g.y0, g.y1);
  const nClasses = Math.max(...d.points.map((p) => p[2])) + 1;

  const pIdx = Math.min(iterIdx, d.perceptronIters.length - 1);
  const pEntry = d.perceptronIters[pIdx];
  const selectedGrid = mode === "perceptron" ? pEntry.grid : d.ridge;
  const title = mode === "perceptron" ? `Perceptron (${pEntry.iters} iters)` : "Ridge Classifier";

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field">
          <div className="seg">
            <button aria-pressed={mode === "perceptron"} onClick={() => setMode("perceptron")}>Perceptron</button>
            <button aria-pressed={mode === "ridge"} onClick={() => setMode("ridge")}>Ridge Classifier</button>
          </div>
        </div>

        {mode === "perceptron" && (
          <div className="field">
            <label>
              Training iterations: <b>{pEntry.iters}</b>
            </label>
            <input
              type="range"
              min={0}
              max={d.perceptronIters.length - 1}
              step={1}
              value={pIdx}
              onChange={(e) => setIterIdx(Number(e.target.value))}
            />
          </div>
        )}
      </div>

      <Chart title={title} caption="Background colors show the class the model predicts in each region; dots are real samples colored by their true class; a dot whose color differs from the region beneath it is a misclassification.">
        <Heat grid={selectedGrid} x0={g.x0} x1={g.x1} y0={g.y0} y1={g.y1} sx={sx} sy={sy} />
        <Points pts={d.points} sx={sx} sy={sy} />
        <Axes x0={g.x0} x1={g.x1} y0={g.y0} y1={g.y1} sx={sx} sy={sy} xlabel="feature 1 (standardized)" ylabel="feature 2 (standardized)" />
      </Chart>

      <div className="legend">
        {Array.from({ length: nClasses }, (_, k) => (
          <LegendItem key={k} color={CLASS_COLORS[k % 3]} dot label={`class ${k}`} tip={CLASS_NAMES[k]} />
        ))}
      </div>

      <p className="callout">
        The Perceptron learns online — its boundary shifts after each misclassification and settles as
        iterations increase; the Ridge Classifier solves regularized least squares in one shot
        (deterministic, no iterations).
      </p>
    </div>
  );
}
