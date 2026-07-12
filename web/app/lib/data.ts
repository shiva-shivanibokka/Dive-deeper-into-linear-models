"use client";
import { useEffect, useState } from "react";

/** Fetch a precomputed artifact from /public once; returns null while loading. */
export function useArtifact<T>(name: string): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    let live = true;
    fetch(`/${name}.json`)
      .then((r) => r.json())
      .then((d) => { if (live) setData(d); })
      .catch(() => { if (live) setData(null); });
    return () => { live = false; };
  }, [name]);
  return data;
}
