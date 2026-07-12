// Dependency-free inline-SVG chart primitives shared by every demo tab.
// Coordinate convention: data space -> screen space via `scale(domain, range)`.
import type { ReactNode } from "react";

export const CLASS_COLORS = ["#22d3ee", "#a78bfa", "#f472b6"]; // class 0,1,2
export const PLOT = { w: 440, h: 280, ml: 48, mr: 16, mt: 16, mb: 40 };

const RX: [number, number] = [PLOT.ml, PLOT.w - PLOT.mr];
const RY: [number, number] = [PLOT.h - PLOT.mb, PLOT.mt]; // inverted (SVG y grows down)

/** Linear mapper from a data domain to a screen range. */
export function scale(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const span = d1 - d0 || 1;
  return (v: number) => range[0] + ((v - d0) / span) * (range[1] - range[0]);
}

export type Scales = { sx: (v: number) => number; sy: (v: number) => number };

/** Build x/y scales from data bounds; returns mappers into the standard plot box. */
export function makeScales(x0: number, x1: number, y0: number, y1: number): Scales {
  return { sx: scale([x0, x1], RX), sy: scale([y0, y1], RY) };
}

export function Chart({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${PLOT.w} ${PLOT.h}`} role="img" aria-label={title}>
        <title>{title}</title>
        {children}
      </svg>
    </div>
  );
}

/** Axes frame with a few tick labels. */
export function Axes({ x0, x1, y0, y1, sx, sy, xlabel, ylabel }: {
  x0: number; x1: number; y0: number; y1: number; sx: (v: number) => number; sy: (v: number) => number;
  xlabel?: string; ylabel?: string;
}) {
  const xticks = [x0, (x0 + x1) / 2, x1];
  const yticks = [y0, (y0 + y1) / 2, y1];
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
  return (
    <g fontFamily="var(--font-mono)" fontSize="9" fill="var(--muted)">
      <line x1={PLOT.ml} y1={PLOT.mt} x2={PLOT.ml} y2={PLOT.h - PLOT.mb} stroke="var(--border)" />
      <line x1={PLOT.ml} y1={PLOT.h - PLOT.mb} x2={PLOT.w - PLOT.mr} y2={PLOT.h - PLOT.mb} stroke="var(--border)" />
      {xticks.map((t, i) => (
        <text key={i} x={sx(t)} y={PLOT.h - PLOT.mb + 14} textAnchor="middle">{fmt(t)}</text>
      ))}
      {yticks.map((t, i) => (
        <text key={i} x={PLOT.ml - 6} y={sy(t) + 3} textAnchor="end">{fmt(t)}</text>
      ))}
      {xlabel && <text x={(PLOT.ml + PLOT.w - PLOT.mr) / 2} y={PLOT.h - 4} textAnchor="middle" fill="var(--muted)">{xlabel}</text>}
      {ylabel && <text x={12} y={(PLOT.mt + PLOT.h - PLOT.mb) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(PLOT.mt + PLOT.h - PLOT.mb) / 2})`} fill="var(--muted)">{ylabel}</text>}
    </g>
  );
}

export function Line({ pts, sx, sy, stroke = "var(--cyan)", width = 2, dash }: {
  pts: [number, number][]; sx: (v: number) => number; sy: (v: number) => number;
  stroke?: string; width?: number; dash?: string;
}) {
  if (!pts.length) return null;
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(2)} ${sy(p[1]).toFixed(2)}`).join(" ");
  return <path d={d} fill="none" stroke={stroke} strokeWidth={width} strokeDasharray={dash} strokeLinejoin="round" />;
}

export function Scatter({ pts, sx, sy, color = "var(--muted)", r = 2.2, opacity = 0.65 }: {
  pts: [number, number][]; sx: (v: number) => number; sy: (v: number) => number;
  color?: string; r?: number; opacity?: number;
}) {
  return <g fill={color} opacity={opacity}>{pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={r} />)}</g>;
}

/** Colored class points: pts are [x, y, class]. */
export function Points({ pts, sx, sy, r = 3 }: {
  pts: [number, number, number][]; sx: (v: number) => number; sy: (v: number) => number; r?: number;
}) {
  return (
    <g>{pts.map((p, i) => (
      <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={r} fill={CLASS_COLORS[p[2] % 3]} stroke="#0b1024" strokeWidth={0.6} />
    ))}</g>
  );
}

/** Shaded band: pts are [x, lo, hi]. */
export function Band({ pts, sx, sy, fill = "var(--violet)", opacity = 0.18 }: {
  pts: [number, number, number][]; sx: (v: number) => number; sy: (v: number) => number;
  fill?: string; opacity?: number;
}) {
  if (!pts.length) return null;
  const top = pts.map((p) => `${sx(p[0]).toFixed(2)} ${sy(p[2]).toFixed(2)}`);
  const bot = pts.slice().reverse().map((p) => `${sx(p[0]).toFixed(2)} ${sy(p[1]).toFixed(2)}`);
  return <path d={`M${top.join(" L")} L${bot.join(" L")} Z`} fill={fill} opacity={opacity} />;
}

/** Decision-boundary heat map from a class grid (rows=y, cols=x). */
export function Heat({ grid, x0, x1, y0, y1, sx, sy, opacity = 0.22 }: {
  grid: number[][]; x0: number; x1: number; y0: number; y1: number;
  sx: (v: number) => number; sy: (v: number) => number; opacity?: number;
}) {
  const rows = grid.length, cols = grid[0]?.length ?? 0;
  if (!rows || !cols) return null;
  const cw = (sx(x1) - sx(x0)) / (cols - 1);
  const ch = (sy(y0) - sy(y1)) / (rows - 1);
  const cells: ReactNode[] = [];
  for (let ri = 0; ri < rows; ri++) {
    for (let ci = 0; ci < cols; ci++) {
      const cx = x0 + ((x1 - x0) * ci) / (cols - 1);
      const cy = y0 + ((y1 - y0) * ri) / (rows - 1);
      cells.push(
        <rect key={`${ri}-${ci}`} x={sx(cx) - cw / 2} y={sy(cy) - ch / 2} width={cw + 0.6} height={ch + 0.6}
          fill={CLASS_COLORS[grid[ri][ci] % 3]} opacity={opacity} shapeRendering="crispEdges" />
      );
    }
  }
  return <g>{cells}</g>;
}

export function Dot({ x, y, sx, sy, color = "var(--lime)", r = 5 }: {
  x: number; y: number; sx: (v: number) => number; sy: (v: number) => number; color?: string; r?: number;
}) {
  return <circle cx={sx(x)} cy={sy(y)} r={r} fill={color} stroke="#0b1024" strokeWidth={1.5} />;
}

export type Ell = { cx: number; cy: number; rx: number; ry: number; angle: number };
export function Ellipse({ e, sx, sy, color }: { e: Ell; sx: (v: number) => number; sy: (v: number) => number; color: string }) {
  // rx/ry are in data units on the x-axis scale; approximate with x-scale factor.
  const k = sx(1) - sx(0);
  return (
    <ellipse cx={sx(e.cx)} cy={sy(e.cy)} rx={Math.abs(e.rx * k)} ry={Math.abs(e.ry * k)}
      transform={`rotate(${-e.angle} ${sx(e.cx)} ${sy(e.cy)})`}
      fill="none" stroke={color} strokeWidth={1.5} strokeDasharray="4 3" opacity={0.9} />
  );
}
