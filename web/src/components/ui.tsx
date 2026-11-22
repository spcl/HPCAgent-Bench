// Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
// The pieces every section shares: its frame and heading, a copyable code block, an arrow link.
import { ArrowUpRight, Check, Copy } from "lucide-react";
import { type ReactNode, useState } from "react";

export function Section({
  id,
  eyebrow,
  title,
  lead,
  aside,
  band = false,
  children,
}: {
  id?: string;
  eyebrow?: string;
  title: string;
  lead?: ReactNode;
  aside?: ReactNode;
  /** A white band across the page instead of the paper background. */
  band?: boolean;
  children: ReactNode;
}) {
  return (
    <section id={id} className={band ? "border-y border-stone-200 bg-white" : ""}>
      <div className="mx-auto max-w-5xl px-5 py-12">
        <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
          <div>
            {eyebrow && <p className="font-mono-num text-xs font-medium tracking-wide text-blue-700 uppercase">{eyebrow}</p>}
            <h2 className="mt-1.5 text-2xl font-bold tracking-tight text-zinc-950">{title}</h2>
            {lead && <p className="mt-2 max-w-3xl text-base leading-relaxed text-zinc-600">{lead}</p>}
          </div>
          {aside}
        </div>
        {children}
      </div>
    </section>
  );
}

export function ArrowLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} className="inline-flex items-center gap-1 text-xs font-medium text-blue-700 hover:underline">
      {children}
      <ArrowUpRight className="h-3.5 w-3.5" />
    </a>
  );
}

export function CopyBlock({ title, text }: { title: string; text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };
  return (
    <div className="rounded-lg border border-stone-200 bg-paper p-5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-lg font-semibold text-zinc-950">{title}</h3>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1.5 rounded border border-stone-200 bg-white px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-stone-50"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="font-mono-num mt-3 overflow-x-auto rounded border border-stone-200 bg-white p-3.5 text-[11px] leading-relaxed text-zinc-800">
        {text}
      </pre>
    </div>
  );
}
