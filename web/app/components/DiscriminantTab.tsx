"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { makeScales, Chart, Axes, Points, Heat, Ellipse, LegendItem, CLASS_COLORS, type Ell } from "../lib/svg";

const CLASS_NAMES = ["Wine cultivar 1", "Wine cultivar 2", "Wine cultivar 3"];

type Grid = { x0: number; x1: number; y0: number; y1: number; step: number };
type Discriminant = {
  points: [number, number, number][];
  grid: Grid;
  lda: number[][];
  qda: number[][];
  ellipses: { lda: Ell[]; qda: Ell[] };
};

export default function DiscriminantTab() {
  const [mode, setMode] = useState<"lda" | "qda">("lda");
  const d = useArtifact<Discriminant>("discriminant");
  if (!d) return <p className="note">loading…</p>;

  const g = d.grid;
  const { sx, sy } = makeScales(g.x0, g.x1, g.y0, g.y1);
  const selectedGrid = mode === "lda" ? d.lda : d.qda;
  const ellipses = mode === "lda" ? d.ellipses.lda : d.ellipses.qda;
  const nClasses = Math.max(...d.points.map((p) => p[2])) + 1;

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field">
          <div className="seg">
            <button aria-pressed={mode === "lda"} onClick={() => setMode("lda")}>LDA</button>
            <button aria-pressed={mode === "qda"} onClick={() => setMode("qda")}>QDA</button>
          </div>
        </div>
      </div>

      <Chart title="LDA vs QDA boundary" caption="Background colors show the class the model predicts in each region; dots are real samples colored by their true class; dashed shapes are each class's covariance ellipse; a dot whose color differs from the region beneath it is a misclassification.">
        <Heat grid={selectedGrid} x0={g.x0} x1={g.x1} y0={g.y0} y1={g.y1} sx={sx} sy={sy} />
        {ellipses.map((e, k) => (
          <Ellipse key={k} e={e} sx={sx} sy={sy} color={CLASS_COLORS[k % 3]} />
        ))}
        <Points pts={d.points} sx={sx} sy={sy} />
        <Axes x0={g.x0} x1={g.x1} y0={g.y0} y1={g.y1} sx={sx} sy={sy} xlabel="feature 1 (standardized)" ylabel="feature 2 (standardized)" />
      </Chart>

      <div className="legend">
        {Array.from({ length: nClasses }, (_, k) => (
          <LegendItem key={k} color={CLASS_COLORS[k % 3]} dot label={`class ${k}`} tip={CLASS_NAMES[k]} />
        ))}
      </div>

      <p className="callout">
        LDA assumes all classes share one covariance → straight boundaries and identical ellipse
        shapes; QDA gives each class its own covariance → curved boundaries and differently-shaped
        ellipses.
      </p>
    </div>
  );
}
