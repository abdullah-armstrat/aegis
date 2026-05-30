// Renders the scorecard: a neutral summary plus a FlagCard per check. Flags that fired are
// shown first so the user sees what to check; clear / not-assessed checks follow, so coverage
// is visible and honest. Never a single trust verdict (the project's spine).

import FlagCard from "./FlagCard";

const STATUS_ORDER = { fired: 0, not_assessed: 1, clear: 2 };

export default function Scorecard({ scorecard }) {
  if (!scorecard) return null;

  const flags = [...(scorecard.flags ?? [])].sort(
    (a, b) => (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9)
  );
  const firedCount = flags.filter((f) => f.status === "fired").length;

  return (
    <section className="mt-8">
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-baseline justify-between">
          <h2 className="text-lg font-semibold text-slate-800">What we found</h2>
          <span className="text-sm text-slate-500">
            {firedCount} flag{firedCount !== 1 ? "s" : ""} raised
          </span>
        </div>
        {scorecard.summary && <p className="mt-1 text-sm text-slate-600">{scorecard.summary}</p>}
        <p className="mt-2 text-xs text-slate-400">
          Aegis points out things worth checking — it does not decide whether the post is true or
          false. The judgement stays with you.
        </p>
      </div>

      <div className="mt-4 space-y-3">
        {flags.map((flag, i) => (
          <FlagCard key={`${flag.type}-${flag.source}-${i}`} flag={flag} />
        ))}
      </div>
    </section>
  );
}
