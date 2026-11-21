import { ArrowUpRight, BookOpen, Github } from "lucide-react";

interface NavbarProps {
  docsUrl: string;
  repoUrl: string;
  paperUrl: string;
}

export function Navbar({ docsUrl, repoUrl, paperUrl }: NavbarProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-stone-200/90 bg-[#FAF9F6]/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-3.5">
        <a href="#" className="flex items-baseline gap-2.5">
          <span className="font-editorial text-xl font-semibold tracking-tight text-zinc-950">
            HPCAgent-Bench
          </span>
          <span className="hidden text-xs text-zinc-400 sm:inline">SPCL · ETH Zurich</span>
        </a>

        <nav className="flex items-center gap-4 text-sm text-zinc-600 sm:gap-5">
          <a href="#leaderboard" className="hidden transition hover:text-zinc-950 sm:inline">
            Models
          </a>
          <a href="#difficulty" className="hidden transition hover:text-zinc-950 md:inline">
            Difficulty
          </a>
          <a href="#ablations" className="hidden transition hover:text-zinc-950 sm:inline">
            Ablations
          </a>
          <a href="#skills" className="hidden transition hover:text-zinc-950 md:inline">
            Skills
          </a>
          <a
            href={paperUrl}
            target="_blank"
            rel="noreferrer"
            className="hidden transition hover:text-zinc-950 lg:inline"
          >
            Paper
          </a>
          <a
            href={docsUrl}
            className="inline-flex items-center gap-1 rounded-md border border-blue-200 bg-blue-50/80 px-2.5 py-1 text-xs font-medium text-blue-800 transition hover:border-blue-300 hover:bg-blue-100/70"
          >
            <BookOpen className="h-3.5 w-3.5" />
            Docs
            <ArrowUpRight className="h-3 w-3" />
          </a>
          <a
            href={repoUrl}
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
            className="inline-flex items-center gap-1 text-zinc-600 transition hover:text-zinc-950"
          >
            <Github className="h-4 w-4" />
            <span className="hidden text-xs font-medium sm:inline">GitHub</span>
          </a>
        </nav>
      </div>
    </header>
  );
}
