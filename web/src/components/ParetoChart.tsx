// Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
import { useMemo, useState } from "react";
import { weightedTokens, type LeaderboardData, type WeightProfileId } from "../types/leaderboard";
import { metricsOf, weightProfiles } from "./ModelPerformanceTable";

interface PlotPoint {
  id: string;
  name: string;
  speedup: number;
  xCost: number;
  success: number;
  isPareto: boolean;
}

export function ParetoChart({
  data,
  activeTrack,
  weightProfile,
}: {
  data: LeaderboardData;
  activeTrack: string;
  weightProfile: WeightProfileId;
}) {
  const [hovered, setHovered] = useState<PlotPoint | null>(null);
  const weights = useMemo(() => weightProfiles(data.token_weights)[weightProfile].weights, [data.token_weights, weightProfile]);

  const { points, paretoLine, maxX, maxY } = useMemo(() => {
    const raw: Omit<PlotPoint, "isPareto">[] = [];
    for (const m of data.models) {
      const tm = metricsOf(m, activeTrack);
      if (tm?.geomean_speedup == null) continue;
      const cost = weightedTokens(tm, weights);
      if (!Number.isFinite(cost)) continue;
      raw.push({ id: m.id, name: m.name, speedup: tm.geomean_speedup, xCost: cost, success: tm.success_rate });
    }

    const byCost = [...raw].sort((a, b) => a.xCost - b.xCost || b.speedup - a.speedup);
    const frontierIds = new Set<string>();
    const frontierPts: PlotPoint[] = [];
    let bestS = -1;
    for (const p of byCost) {
      if (p.speedup > bestS) {
        bestS = p.speedup;
        frontierIds.add(p.id);
        frontierPts.push({ ...p, isPareto: true });
      }
    }

    return {
      points: raw.map((p) => ({ ...p, isPareto: frontierIds.has(p.id) })),
      paretoLine: frontierPts,
      maxX: Math.max(...raw.map((p) => p.xCost), 1) * 1.2,
      maxY: Math.max(...raw.map((p) => p.speedup), 2) * 1.18,
    };
  }, [data.models, activeTrack, weights]);

  const width = 760;
  const height = 250;
  const pad = { top: 20, right: 32, bottom: 42, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xScale = (v: number) => pad.left + (v / maxX) * plotW;
  const yScale = (v: number) => pad.top + plotH - (v / maxY) * plotH;
  const poly = paretoLine.map((p) => `${xScale(p.xCost)},${yScale(p.speedup)}`).join(" ");

  return (
    <section className="mx-auto max-w-5xl px-5 pb-12">
      <div className="rounded-lg border border-stone-200 bg-white p-5">
        <div className="flex flex-col justify-between gap-1 sm:flex-row sm:items-center">
          <div>
            <h3 className="font-editorial text-xl font-medium text-zinc-950">
              Speedup against token cost
            </h3>
            <p className="text-xs text-zinc-500">
              {`Weighted tokens per task, arithmetic mean: ${weights.in}·fresh + ${weights.cache}·cached + ${weights.out}·output (M).`}
            </p>
          </div>
          <span className="font-mono-num text-[11px] text-blue-700">
            Dashed curve = Pareto frontier
          </span>
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} className="mt-3 w-full" role="img" aria-label="Pareto frontier chart">
          {[0, 0.25, 0.5, 0.75, 1].map((f) => {
            const val = f * maxY;
            const y = yScale(val);
            return (
              <g key={f}>
                <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} stroke="#E7E5E4" strokeDasharray="3 3" />
                <text x={pad.left - 8} y={y + 3.5} textAnchor="end" fontSize="10" fill="#71717A" className="font-mono-num">
                  {val.toFixed(1)}×
                </text>
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => {
            const val = f * maxX;
            const x = xScale(val);
            return (
              <g key={f}>
                <line x1={x} y1={pad.top} x2={x} y2={pad.top + plotH} stroke="#F5F5F4" />
                <text x={x} y={height - 19} textAnchor="middle" fontSize="10" fill="#71717A" className="font-mono-num">
                  {`${val.toFixed(1)}M`}
                </text>
              </g>
            );
          })}
          <text x={pad.left + plotW / 2} y={height - 3} textAnchor="middle" fontSize="11" fill="#52525B">
            Weighted tokens per task (M, lower is cheaper)
          </text>
          <text
            x={15}
            y={pad.top + plotH / 2}
            textAnchor="middle"
            fontSize="11"
            fill="#52525B"
            transform={`rotate(-90 15 ${pad.top + plotH / 2})`}
          >
            Speedup (geomean)
          </text>
          {paretoLine.length > 1 && (
            <polyline fill="none" stroke="#1D4ED8" strokeWidth="1.6" strokeDasharray="5 4" opacity="0.75" points={poly} />
          )}
          {points.map((p) => {
            const cx = xScale(p.xCost);
            const cy = yScale(p.speedup);
            const isH = hovered?.id === p.id;
            return (
              <g
                key={p.id}
                onMouseEnter={() => setHovered(p)}
                onMouseLeave={() => setHovered(null)}
                className="cursor-pointer"
              >
                {p.isPareto && (
                  <circle cx={cx} cy={cy} r={isH ? 12 : 9.5} fill="none" stroke="#1D4ED8" strokeWidth="1.3" opacity="0.35" />
                )}
                <circle cx={cx} cy={cy} r={isH ? 7 : 5.5} fill="#1D4ED8" stroke="#FFFFFF" strokeWidth="1.5" />
                <text x={cx + 9} y={cy - 6} fontSize="10.5" fill="#18181B" fontWeight="600">
                  {p.name}
                </text>
                <text x={cx + 9} y={cy + 7} fontSize="9.5" fill="#71717A" className="font-mono-num">
                  {p.speedup.toFixed(2)}× · {p.xCost.toFixed(1)}M
                </text>
              </g>
            );
          })}
        </svg>

        {hovered && (
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 rounded border border-stone-200 bg-paper px-3.5 py-2 text-xs text-zinc-700">
            <strong className="text-zinc-950">{hovered.name}</strong>
            <span className="font-mono-num">
              Speedup <strong>{hovered.speedup.toFixed(2)}×</strong> · solved{" "}
              <strong>{(hovered.success * 100).toFixed(0)}%</strong> · tokens{" "}
              <strong>{hovered.xCost.toFixed(2)}M/task</strong>
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
