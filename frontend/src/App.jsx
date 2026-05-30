import { useState } from "react";
import { analyze } from "./api";
import SubmitPanel from "./components/SubmitPanel";
import Scorecard from "./components/Scorecard";

export default function App() {
  const [scorecard, setScorecard] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAnalyze(file, caption) {
    setLoading(true);
    setError(null);
    setScorecard(null);
    try {
      setScorecard(await analyze(file, caption));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <div className="mx-auto max-w-2xl px-6 py-12">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">Aegis</h1>
          <p className="mt-2 text-slate-600">
            A cross-consistency auditor for image posts. It surfaces contradictions and recycled
            context and explains them — so you can judge for yourself. It never gives a trust
            verdict.
          </p>
        </header>

        <main className="mt-8">
          <SubmitPanel onAnalyze={handleAnalyze} loading={loading} />

          {error && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <Scorecard scorecard={scorecard} />
        </main>
      </div>
    </div>
  );
}
