import { useEffect, useState } from "react";
import { getHealth } from "./api";

// Week-0 skeleton: proves the React -> FastAPI cross-origin path is wired.
// SubmitPanel / Scorecard / FlagCard components replace this in Week 2.
export default function App() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(e.message));
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <div className="mx-auto max-w-2xl px-6 py-16">
        <h1 className="text-3xl font-bold tracking-tight">Aegis</h1>
        <p className="mt-2 text-slate-600">
          A multi-modal cross-consistency auditor. It surfaces cross-modal contradictions and
          recycled context, and explains them — it never gives a trust verdict.
        </p>

        <div className="mt-8 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            Backend status
          </h2>
          {error && <p className="mt-2 text-red-600">Cannot reach backend: {error}</p>}
          {!error && !health && <p className="mt-2 text-slate-400">Checking…</p>}
          {health && (
            <pre className="mt-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100">
              {JSON.stringify(health, null, 2)}
            </pre>
          )}
        </div>

        <p className="mt-6 text-sm text-slate-400">
          Scaffold only. Submit panel and scorecard arrive in Week 2.
        </p>
      </div>
    </div>
  );
}
