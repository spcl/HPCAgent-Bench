// Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
import type { PaperMeta } from "../types/leaderboard";

export function Footer({ paper }: { paper: PaperMeta }) {
  return (
    <footer className="border-t border-stone-200 bg-paper py-8">
      <div className="mx-auto flex max-w-5xl flex-col items-center justify-between gap-3 px-5 text-xs text-zinc-500 sm:flex-row">
        <div>
          <span className="text-sm font-semibold text-zinc-800">HPCAgent-Bench</span> · SPCL, ETH Zurich ·{" "}
          {paper.license}
        </div>
        <div className="flex items-center gap-4">
          <a href={paper.docs_url} className="hover:text-zinc-900">
            Documentation
          </a>
          <a href={paper.arxiv_url} target="_blank" rel="noreferrer" className="hover:text-zinc-900">
            arXiv:{paper.arxiv_id}
          </a>
          <a href={paper.repo_url} target="_blank" rel="noreferrer" className="hover:text-zinc-900">
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
