"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";
import { Chart, makeScales, Axes, Line } from "../lib/svg";

type Model = "ridge" | "lasso" | "elastic";
type Art = {
  features: string[];
  alphas: number[];
  ridge: number[][];
  lasso: number[][];
  elastic: number[][];
};

const COLORS = [
  "var(--cyan)", "var(--violet)", "var(--pink)", "var(--lime)", "var(--amber)",
  "var(--ok)", "var(--bad)", "#60a5fa", "#f59e0b", "#34d399",
];

export default function RegularizationTab() {
  const d = useArtifact<Art>("regularization");
  const [model, setModel] = useState<Model>("ridge");
  const [i, setI] = useState(0);
  if (!d) return <p className="note">loading…</p>;

  const coefs = d[model];
  const k = Math.min(i, d.alphas.length - 1);
  const logs = d.alphas.map((a) => Math.log10(a));
  const x0 = Math.min(...logs), x1 = Math.max(...logs);

  const flat = coefs.flat();
  let y0 = Math.min(...flat), y1 = Math.max(...flat);
  const pad = (y1 - y0) * 0.05 || 1;
  y0 -= pad; y1 += pad;

  const { sx, sy } = makeScales(x0, x1, y0, y1);
  const marker = Math.log10(d.alphas[k]);

  const atAlpha = d.features.map((_, j) => coefs[k][j]);
  const maxAbs = Math.max(...atAlpha.map(Math.abs)) || 1;

  return (
    <div className="demo">
      <div className="control-row">
        <div className="field">
          <label><span className="lname">Model</span></label>
          <div className="seg">
            <button aria-pressed={model === "ridge"} onClick={() => setModel("ridge")}>Ridge</button>
            <button aria-pressed={model === "lasso"} onClick={() => setModel("lasso")}>Lasso</button>
            <button aria-pressed={model === "elastic"} onClick={() => setModel("elastic")}>ElasticNet</button>
          </div>
        </div>
        <div className="field" style={{ maxWidth: 320 }}>
          <label htmlFor="alpha"><span className="lname">Penalty α: <b>{d.alphas[k].toFixed(3)}</b></span></label>
          <input id="alpha" type="range" min={0} max={d.alphas.length - 1} value={k}
            onChange={(e) => setI(Number(e.target.value))} />
        </div>
      </div>

      <Chart title="Coefficient paths">
        <Axes x0={x0} x1={x1} y0={y0} y1={y1} sx={sx} sy={sy} xlabel="log₁₀(α)" ylabel="coefficient" />
        {d.features.map((f, j) => (
          <Line key={f} pts={d.alphas.map((a, ai) => [Math.log10(a), coefs[ai][j]])}
            sx={sx} sy={sy} stroke={COLORS[j % COLORS.length]} width={1.8} />
        ))}
        <Line pts={[[marker, y0], [marker, y1]]} sx={sx} sy={sy} stroke="var(--muted)" width={1.5} dash="3 3" />
      </Chart>

      <div className="legend">
        {d.features.map((f, j) => (
          <span key={f} className="item">
            <span className="swatch" style={{ background: COLORS[j % COLORS.length] }} />{f}
          </span>
        ))}
      </div>

      <div className="bars">
        {d.features.map((f, j) => (
          <div key={f} className="bar-row">
            <span className="name">{f}</span>
            <div className="bar-track">
              <div className="fill" style={{ width: `${(Math.abs(atAlpha[j]) / maxAbs) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>

      <p className="callout">
        A bigger <strong>α</strong> shrinks the coefficients. <strong>Lasso (L1)</strong> drives some to exactly
        zero — that is built-in feature selection — while <strong>Ridge (L2)</strong> shrinks them smoothly toward
        zero without eliminating any. <strong>ElasticNet</strong> blends both behaviors.
      </p>
    </div>
  );
}
