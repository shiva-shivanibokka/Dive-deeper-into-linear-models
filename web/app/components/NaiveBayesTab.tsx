"use client";
import { useArtifact } from "../lib/data";
import { makeScales, Chart, Axes, Points, Heat, CLASS_COLORS } from "../lib/svg";

type Grid = { x0: number; x1: number; y0: number; y1: number; step: number };
type NaiveBayes = {
  points: [number, number, number][];
  grid: Grid;
  boundary: number[][];
};

export default function NaiveBayesTab() {
  const d = useArtifact<NaiveBayes>("naivebayes");
  if (!d) return <p className="note">loading…</p>;

  const g = d.grid;
  const { sx, sy } = makeScales(g.x0, g.x1, g.y0, g.y1);
  const nClasses = Math.max(...d.points.map((p) => p[2])) + 1;

  return (
    <div className="demo">
      <Chart title="Gaussian Naive Bayes regions">
        <Heat grid={d.boundary} x0={g.x0} x1={g.x1} y0={g.y0} y1={g.y1} sx={sx} sy={sy} />
        <Points pts={d.points} sx={sx} sy={sy} />
        <Axes x0={g.x0} x1={g.x1} y0={g.y0} y1={g.y1} sx={sx} sy={sy} xlabel="feature 1 (standardized)" ylabel="feature 2 (standardized)" />
      </Chart>

      <div className="legend">
        {Array.from({ length: nClasses }, (_, k) => (
          <div className="item" key={k}>
            <span className="dot" style={{ background: CLASS_COLORS[k % 3] }} />
            class {k}
          </div>
        ))}
      </div>

      <p className="callout">
        Naive Bayes assumes the two features are independent within each class; the regions are where
        each class&apos;s Gaussian is most probable.
      </p>
    </div>
  );
}
