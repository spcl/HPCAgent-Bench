// Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
import { useEffect, useState } from "react";
import { ABLATIONS, MODELS, PAPER, TRACKS } from "../content";
import type { ExportBoard, ExportScore, LeaderboardData, ModelEntry, TrackMetrics } from "../types/leaderboard";

const MILLION = 1e6;

function trackMetrics(score: ExportScore): TrackMetrics {
  return {
    geomean_speedup: score.geomean,
    ci_low: score.ci_low,
    ci_high: score.ci_high,
    success_rate: score.kernels ? score.solved / score.kernels : 0,
    kernels: score.kernels,
    mean_t_in_m: score.fresh_input_per_task / MILLION,
    mean_t_cache_m: score.cached_input_per_task / MILLION,
    mean_t_out_m: score.output_per_task / MILLION,
    per_kernel: score.per_kernel,
  };
}

/** One entry per control setup (no condition): the score panels compare models, the ablations compare conditions. */
export function modelsOf(board: ExportBoard): ModelEntry[] {
  const bySetup = new Map<string, ExportScore[]>();
  for (const score of board.scores.filter((s) => s.condition === "")) {
    bySetup.set(score.setup, [...(bySetup.get(score.setup) ?? []), score]);
  }
  return [...bySetup.entries()].map(([setup, scores]) => {
    const token = scores[0].model;
    const meta = MODELS[token] ?? { name: token, provider: "", harness: "", params_label: "" };
    const tracks: Record<string, TrackMetrics> = {};
    for (const track of TRACKS) {
      const score = scores.find((s) => s.track === track.track);
      if (score) tracks[track.id] = trackMetrics(score);
    }
    return { ...meta, id: token, setup, tracks };
  });
}

export function useLeaderboard() {
  const [data, setData] = useState<LeaderboardData | null>(null);
  useEffect(() => {
    let active = true;
    const fixed = { paper: PAPER, tracks: TRACKS, ablations: ABLATIONS };
    fetch("./data/leaderboard.json")
      .then((res) => (res.ok ? (res.json() as Promise<ExportBoard>) : null))
      .catch(() => null)
      .then((board) => {
        if (!active) return;
        const usable = board !== null && board.schema_version === 1 ? board : null;
        const card = usable?.cost ?? { fresh_input: 1, cached_input: 0.1, output: 1 };
        setData({
          ...fixed,
          board: usable,
          models: usable ? modelsOf(usable) : [],
          token_weights: { in: card.fresh_input, cache: card.cached_input, out: card.output },
        });
      });
    return () => {
      active = false;
    };
  }, []);
  return data;
}
