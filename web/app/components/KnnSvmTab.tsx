"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { makeScales, Chart, Axes, Points, Heat, LegendItem, CLASS_COLORS } from "../lib/svg";

const CLASS_NAMES = ["Iris setosa", "Iris versicolor", "Iris virginica"];

type Grid = { x0: number; x1: number; y0: number; y1: number; step: number };
type KnnSvm = {
  points: [number, number, number][];
  grid: Grid;
  knn: { k: number; grid: number[][] }[];
  svm: { kernel: string; C: number; gamma: string; grid: number[][] }[];
};

export default function KnnSvmTab() {
  const [mode, setMode] = useState<"knn" | "svm">("knn");
  const [kIdx, setKIdx] = useState(0);
  const [svmIdx, setSvmIdx] = useState(0);
  const d = useArtifact<KnnSvm>("knnsvm");
  if (!d) return <p className="note">loading…</p>;

  const g = d.grid;
  const { sx, sy } = makeScales(g.x0, g.x1, g.y0, g.y1);
  const nClasses = Math.max(...d.points.map((p) => p[2])) + 1;

  const knnEntry = d.knn[Math.min(kIdx, d.knn.length - 1)];
  const svmEntry = d.svm[Math.min(svmIdx, d.svm.length - 1)];
  const selectedGrid = mode === "knn" ? knnEntry.grid : svmEntry.grid;
  const title = mode === "knn" ? `k-NN (k=${knnEntry.k})` : `SVM (${svmEntry.kernel})`;

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field">
          <div className="seg">
            <button aria-pressed={mode === "knn"} onClick={() => setMode("knn")}>KNN</button>
            <button aria-pressed={mode === "svm"} onClick={() => setMode("svm")}>SVM</button>
          </div>
        </div>

        {mode === "knn" ? (
          <div className="field">
            <div className="seg">
              {d.knn.map((e, i) => (
                <button key={e.k} aria-pressed={i === kIdx} onClick={() => setKIdx(i)}>k={e.k}</button>
              ))}
            </div>
          </div>
        ) : (
          <div className="field">
            <div className="seg">
              {d.svm.map((e, i) => (
                <button key={e.kernel} aria-pressed={i === svmIdx} onClick={() => setSvmIdx(i)}>{e.kernel}</button>
              ))}
            </div>
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
        k-NN draws boundaries by local majority vote (small k = jagged); SVM finds a maximum-margin
        boundary, straight for the linear kernel and curved for rbf/poly.
      </p>
    </div>
  );
}
