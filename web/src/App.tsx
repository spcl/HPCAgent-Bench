// Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
import { useState } from "react";
import { AblationStudySection } from "./components/AblationStudySection";
import { DifficultyAgreementSection } from "./components/DifficultyAgreementSection";
import { DocsAndQuickstart } from "./components/DocsAndQuickstart";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { ModelPerformanceTable } from "./components/ModelPerformanceTable";
import { Navbar } from "./components/Navbar";
import { ParetoChart } from "./components/ParetoChart";
import { SkillsShowcaseSection } from "./components/SkillsShowcaseSection";
import { useLeaderboard } from "./hooks/useLeaderboard";
import type { WeightProfileId } from "./types/leaderboard";

export default function App() {
  const data = useLeaderboard();
  const [activeTrack, setActiveTrack] = useState("overall");
  const [weightProfile, setWeightProfile] = useState<WeightProfileId>("export_card");

  if (!data) {
    return <div className="flex min-h-screen items-center justify-center text-sm text-zinc-500">Loading…</div>;
  }
  return (
    <div className="min-h-screen bg-paper text-zinc-900">
      <Navbar paper={data.paper} />
      <main>
        <Hero data={data} />
        <ModelPerformanceTable
          data={data}
          activeTrack={activeTrack}
          onTrackChange={setActiveTrack}
          weightProfile={weightProfile}
          onWeightProfileChange={setWeightProfile}
        />
        <ParetoChart data={data} activeTrack={activeTrack} weightProfile={weightProfile} />
        <DifficultyAgreementSection tracks={data.tracks} />
        <AblationStudySection ablations={data.ablations} docsUrl={data.paper.docs_url} />
        <SkillsShowcaseSection ablations={data.ablations} />
        <DocsAndQuickstart paper={data.paper} />
      </main>
      <Footer paper={data.paper} />
    </div>
  );
}
