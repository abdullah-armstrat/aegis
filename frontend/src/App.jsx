import { useState } from "react";
import { analyze } from "./api";
import SubmitPanel from "./components/SubmitPanel";
import Scorecard from "./components/Scorecard";

export default function App() {
  const [screen, setScreen] = useState("submit"); // submit | scorecard
  const [scorecard, setScorecard] = useState(null);
  const [submission, setSubmission] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAnalyze(file, caption, ctx) {
    setLoading(true);
    setError(null);
    setScorecard(null);
    try {
      const card = await analyze(file, caption, ctx?.postedDate);
      setSubmission({ ...ctx, caption });
      setScorecard(card);
      setScreen("scorecard");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const tab = (id, label) => (
    <button
      type="button"
      onClick={() => setScreen(id)}
      className="border-b-2 pb-1 text-sm transition-colors"
      style={{
        fontWeight: screen === id ? 600 : 500,
        color: screen === id ? "#161616" : "#8C8C8C",
        borderColor: screen === id ? "#161616" : "transparent",
      }}
    >
      {label}
    </button>
  );

  return (
    <div className="min-h-screen bg-white text-ink">
      <div className="mx-auto max-w-4xl px-8">
        {/* Masthead */}
        <header
          className="flex items-center justify-between py-6"
          style={{ borderBottom: "1px solid #E4E4E4" }}
        >
          <div className="flex items-center gap-3.5">
            <div className="h-[22px] w-[22px] rotate-45 bg-ink" />
            <div>
              <div className="text-[23px] font-bold leading-none tracking-[0.02em] text-ink">
                AEGIS
              </div>
              <div className="mt-1.5 text-xs text-muted">
                Image, video and caption misinformation auditor
              </div>
            </div>
          </div>
          <nav className="flex gap-6">
            {tab("submit", "Submit")}
            {tab("scorecard", "Scorecard")}
          </nav>
        </header>

        <main>
          {error && (
            <div
              className="mt-8 bg-white px-4 py-3 font-mono text-[13px] text-ink"
              style={{ border: "1px solid #E4E4E4", borderLeft: "3px solid #B7791F" }}
            >
              Audit could not run: {error}
            </div>
          )}

          {screen === "submit" && (
            <SubmitPanel onAnalyze={handleAnalyze} loading={loading} />
          )}

          {screen === "scorecard" &&
            (scorecard ? (
              <div className="py-12">
                <Scorecard scorecard={scorecard} submission={submission} />
                <button
                  type="button"
                  onClick={() => setScreen("submit")}
                  className="mt-9 px-5 py-2.5 text-sm font-semibold tracking-tight text-ink transition-colors hover:bg-ink hover:text-white"
                  style={{ border: "1px solid #161616" }}
                >
                  ← Review another post
                </button>
              </div>
            ) : (
              <div className="py-24 text-center">
                <p className="text-sm text-muted">
                  No audit yet. Submit a post for review to see its scorecard.
                </p>
                <button
                  type="button"
                  onClick={() => setScreen("submit")}
                  className="mt-4 px-5 py-2.5 text-sm font-semibold tracking-tight text-ink"
                  style={{ border: "1px solid #161616" }}
                >
                  Go to submit
                </button>
              </div>
            ))}
        </main>
      </div>
    </div>
  );
}
