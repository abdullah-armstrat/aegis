import { useState } from "react";
import { analyze } from "./api";
import SubmitPanel from "./components/SubmitPanel";
import Scorecard from "./components/Scorecard";

export default function App() {
  const [scorecard, setScorecard] = useState(null);
  const [submission, setSubmission] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAnalyze(file, caption, ctx) {
    setLoading(true);
    setError(null);
    setScorecard(null);
    try {
      const card = await analyze(file, caption);
      setSubmission({ ...ctx, caption });
      setScorecard(card);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-white text-ink">
      <div className="mx-auto max-w-3xl px-6 py-10">
        {/* Masthead */}
        <header className="flex items-center justify-between border-b border-ink pb-4">
          <div className="flex items-center gap-3">
            {/* Mark: a quiet shield/diamond, echoing the design's wordmark. */}
            <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
              <rect
                x="6"
                y="2.5"
                width="13"
                height="13"
                transform="rotate(45 11 9)"
                fill="#161616"
              />
            </svg>
            <span className="text-lg font-bold uppercase tracking-[0.3em] text-ink">Aegis</span>
          </div>
          <span className="hidden font-mono text-[11px] uppercase tracking-[0.14em] text-muted sm:block">
            Image &amp; caption misinformation auditor
          </span>
        </header>

        <main className="py-10">
          {!scorecard && <SubmitPanel onAnalyze={handleAnalyze} loading={loading} />}

          {error && (
            <div className="mt-6 border border-flag/50 bg-flagbg px-4 py-3 font-mono text-[13px] text-ink">
              Audit could not run: {error}
            </div>
          )}

          {scorecard && (
            <>
              <Scorecard scorecard={scorecard} submission={submission} />
              <button
                type="button"
                onClick={() => {
                  setScorecard(null);
                  setError(null);
                }}
                className="mt-8 border border-ink px-5 py-2.5 text-sm font-semibold tracking-tight text-ink transition-colors hover:bg-ink hover:text-white"
              >
                ← Review another post
              </button>
            </>
          )}
        </main>

        <footer className="border-t border-line pt-4">
          <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
            Aegis · Explain, don’t verdict · findings are leads, not rulings
          </p>
        </footer>
      </div>
    </div>
  );
}
