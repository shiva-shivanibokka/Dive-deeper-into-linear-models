import { NextResponse } from "next/server";
import model from "./model.json";

// Real inference from a Ridge model trained offline in Python (see
// scripts/export_web_artifacts.py). We ship only the coefficients + standardizer,
// so this runs as a lightweight serverless function with no Python at runtime.
export const runtime = "nodejs";

type Model = {
  features: string[];
  means: number[];
  stds: number[];
  coef: number[];
  intercept: number;
  target: string;
  example: Record<string, number>;
};

const M = model as Model;

function predict(features: Record<string, number>): number {
  let y = M.intercept;
  M.features.forEach((name, i) => {
    const raw = features[name];
    const v = typeof raw === "number" && Number.isFinite(raw) ? raw : M.means[i];
    const z = (v - M.means[i]) / M.stds[i]; // standardize with training stats
    y += M.coef[i] * z;
  });
  return y;
}

// GET -> self-documenting schema so the endpoint is discoverable.
export function GET() {
  return NextResponse.json({
    model: "Ridge regression (California Housing)",
    target: M.target,
    features: M.features,
    example: M.example,
    usage: "POST { features: { MedInc: 5.0, HouseAge: 30, ... } } -> { prediction }",
  });
}

export async function POST(request: Request) {
  let body: { features?: Record<string, number> };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const features = body?.features;
  if (!features || typeof features !== "object") {
    return NextResponse.json({ error: "body must be { features: {...} }" }, { status: 400 });
  }
  const value = predict(features);
  return NextResponse.json({
    prediction: Math.round(value * 1000) / 1000,
    unit: M.target,
    usd: Math.round(value * 100000),
  });
}
