import { ArrowUpRight } from "lucide-react";
import type { LeaderboardData } from "../types/leaderboard";

export function DifficultyAgreementSection({ data }: { data: LeaderboardData }) {
  return (
    <section id="difficulty" className="border-y border-stone-200 bg-white">
      <div className="mx-auto max-w-5xl px-5 py-12">
        <p className="font-mono-num text-xs font-medium tracking-wide text-blue-700 uppercase">
          Core Finding 1 · Why Single-Track Benchmarks Mislead
        </p>
        <h2 className="font-editorial mt-1.5 text-2xl font-medium text-zinc-950">
          Models agree on which loop kernels are hard — not on scientific computing or ML.
        </h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-zinc-600">
          When the same three models run under one judge and one statistical protocol across all
          three tracks, kernel difficulty ceases to be a universal property. A single-track
          benchmark mistakes domain-specific hardness for a general fact about coding agents.
        </p>

        {/* Visual Rank-Correlation Range Bars */}
        <div className="mt-7 grid gap-5 md:grid-cols-3">
          {data.tracks.map((t) => {
            const leftPct = Math.max(0, Math.min(100, t.rank_corr_low * 100));
            const widthPct = Math.max(6, (t.rank_corr_high - t.rank_corr_low) * 100);
            return (
              <article
                key={t.id}
                className="flex flex-col justify-between rounded-lg border border-stone-200 bg-[#FAF9F6] p-5"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span
                      className="font-mono-num text-xs font-semibold uppercase"
                      style={{ color: t.accent_hex }}
                    >
                      Track · {t.short_name}
                    </span>
                    <span className="font-mono-num text-xs text-zinc-400">
                      {t.kernel_count} kernels
                    </span>
                  </div>

                  <h3 className="font-editorial mt-2 text-xl font-medium text-zinc-950">{t.name}</h3>

                  {/* Spearman Rank Correlation Gauge */}
                  <div className="mt-4 rounded-md border border-stone-200 bg-white p-3">
                    <div className="flex items-baseline justify-between">
                      <span className="text-[11px] font-medium text-zinc-500">
                        Inter-model difficulty ρ
                      </span>
                      <span className="font-mono-num text-sm font-semibold text-zinc-950">
                        {t.rank_corr_range}
                      </span>
                    </div>
                    <div className="relative mt-2 h-2.5 w-full rounded-full bg-stone-100">
                      <div
                        className="absolute top-0 h-full rounded-full"
                        style={{
                          left: `${leftPct}%`,
                          width: `${widthPct}%`,
                          backgroundColor: t.accent_hex,
                        }}
                      />
                    </div>
                    <div className="font-mono-num mt-1 flex justify-between text-[10px] text-zinc-400">
                      <span>0.0 (disagree)</span>
                      <span>1.0 (identical)</span>
                    </div>
                    <p className="mt-2 text-[11px] leading-snug text-zinc-600">{t.agreement_note}</p>
                  </div>
                </div>

                <div className="mt-4 border-t border-stone-200/80 pt-3 text-[11px] text-zinc-500">
                  <div>
                    Baseline: <span className="font-mono-num text-zinc-800">{t.baseline_rule}</span>
                  </div>
                  <a
                    href={t.docs_href}
                    className="mt-1.5 inline-flex items-center gap-1 font-medium text-blue-700 hover:underline"
                  >
                    Track guide
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </a>
                </div>
              </article>
            );
          })}
        </div>

        {/* Ranking Reversal Callout */}
        <div className="mt-6 rounded-lg border border-stone-200 bg-[#FAF9F6] px-5 py-4 text-xs leading-relaxed text-zinc-700">
          <strong className="text-zinc-950">Model rankings flip across tracks and GPU scale:</strong>{" "}
          On LLR in C, <span className="font-mono-num font-medium">Kimi-K2.7</span> leads{" "}
          <span className="font-mono-num font-medium">Qwen3.8-27B</span> (Qwen/Kimi ratio{" "}
          <span className="font-mono-num">0.71×</span>). On 4-GPU ML,{" "}
          <span className="font-mono-num font-medium">Qwen3.8-27B</span> reverses the lead by{" "}
          <span className="font-mono-num">2.31×</span>. Within ML, 1-GPU rankings predict 4-GPU
          single-node rankings (<span className="font-mono-num">0.92–0.98</span>) but break down at
          16 GPUs across 4 nodes (<span className="font-mono-num">0.31</span> Kimi-K2.7,{" "}
          <span className="font-mono-num">0.44</span> Qwen3.8-27B).
        </div>
      </div>
    </section>
  );
}
