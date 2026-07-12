"use client";
import { useState } from "react";
import { useArtifact } from "../lib/data";

type Reg = { model: string; dataset: string; r2: number; rmse: number; mae: number };
type Clf = { model: string; dataset: string; accuracy: number; f1: number; rocauc: number };
type Art = { regression: Reg[]; classification: Clf[] };

type Col<T> = {
  key: keyof T & string;
  label: string;
  numeric: boolean;
  better?: "max" | "min";
  fmt?: (v: number) => string;
};

function best<T>(rows: T[], key: keyof T, dir: "max" | "min"): number {
  const vals = rows.map((r) => r[key] as number);
  return dir === "max" ? Math.max(...vals) : Math.min(...vals);
}

function SortableTable<T>({ rows, cols, initial }: { rows: T[]; cols: Col<T>[]; initial: keyof T & string }) {
  const [sortKey, setSortKey] = useState<keyof T & string>(initial);
  // Stable, numeric, always-descending sort (index tiebreak keeps ties in source order).
  const sorted = rows
    .map((row, i) => ({ row, i }))
    .sort((a, b) => {
      const diff = (b.row[sortKey] as number) - (a.row[sortKey] as number);
      return diff !== 0 ? diff : a.i - b.i;
    })
    .map((x) => x.row);

  const bestOf: Partial<Record<keyof T & string, number>> = {};
  for (const c of cols) {
    if (c.numeric && c.better) bestOf[c.key] = best(rows, c.key, c.better);
  }

  return (
    <table className="cmp-table">
      <thead>
        <tr>
          {cols.map((c) => (
            <th
              key={c.key}
              aria-sort={c.numeric && sortKey === c.key ? "descending" : undefined}
              onClick={c.numeric ? () => setSortKey(c.key) : undefined}
              style={c.numeric ? { cursor: "pointer" } : undefined}
            >
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map((row, ri) => (
          <tr key={ri}>
            {cols.map((c) => {
              const raw = row[c.key];
              const isBest = c.numeric && bestOf[c.key] === (raw as number);
              return (
                <td key={c.key} className={isBest ? "best" : undefined}>
                  {c.numeric && c.fmt ? c.fmt(raw as number) : String(raw)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const f3 = (v: number) => v.toFixed(3);
const f2 = (v: number) => v.toFixed(2);

export default function ComparisonTab() {
  const d = useArtifact<Art>("comparison");
  if (!d) return <p className="note">loading…</p>;

  const regCols: Col<Reg>[] = [
    { key: "model", label: "Model", numeric: false },
    { key: "dataset", label: "Dataset", numeric: false },
    { key: "r2", label: "R²", numeric: true, better: "max", fmt: f3 },
    { key: "rmse", label: "RMSE", numeric: true, better: "min", fmt: f2 },
    { key: "mae", label: "MAE", numeric: true, better: "min", fmt: f2 },
  ];
  const clfCols: Col<Clf>[] = [
    { key: "model", label: "Model", numeric: false },
    { key: "dataset", label: "Dataset", numeric: false },
    { key: "accuracy", label: "Accuracy", numeric: true, better: "max", fmt: f3 },
    { key: "f1", label: "F1", numeric: true, better: "max", fmt: f3 },
    { key: "rocauc", label: "ROC-AUC", numeric: true, better: "max", fmt: f3 },
  ];

  const topReg = d.regression.reduce((a, b) => (b.r2 > a.r2 ? b : a));
  const topClf = d.classification.reduce((a, b) => (b.accuracy > a.accuracy ? b : a));

  return (
    <div className="demo">
      <div className="section-label">Regression — Diabetes</div>
      <SortableTable rows={d.regression} cols={regCols} initial="r2" />

      <div className="section-label">Classification — Breast Cancer</div>
      <SortableTable rows={d.classification} cols={clfCols} initial="accuracy" />

      <p className="callout">
        No single model wins every metric — that is the whole point of benchmarking on a shared train/test split.
        Here <strong>{topReg.model}</strong> tops the regression table on R², while <strong>{topClf.model}</strong>
        {" "}leads the classifiers on accuracy, yet each can trail on another column. Compare across metrics before
        you commit.
      </p>
    </div>
  );
}
