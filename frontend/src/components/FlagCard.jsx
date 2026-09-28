// One finding as a card: white, with a thin border and a left rail (icon box and state label).
// Colour is only a small accent (the 42px icon box, the state label, an outlined badge).
// Shows the backend Flag:
//   { type, status, severity, evidence, plain_explanation, what_to_check, source }
// The card explains; it does not give a verdict. not_assessed looks different from clear, so a
// check that could not run never looks like a pass.

import Signal, { STATUS_META } from "./Signal";

const TYPE_LABELS = {
  caption_content_mismatch: "Caption ↔ picture match",
  recycled_context: "Recycled context",
  emotional_framing: "Shouting style",
  audio_visual_mismatch: "Speech ↔ picture match",
};

// Seconds as m:ss, the way the backend's evidence cites moments.
export function clock(seconds) {
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

// Backend severity (info|low|medium|high) shown as a plain word.
const SEVERITY_WORD = { high: "Notable", medium: "Notable", low: "Minor", info: "Minor" };

const CHECK_HEADING = {
  fired: "What to check",
  clear: "What to check",
  not_assessed: "What to check yourself",
};

export default function FlagCard({ flag, onSeek }) {
  const meta = STATUS_META[flag.status] ?? STATUS_META.not_assessed;
  const typeLabel = TYPE_LABELS[flag.type] ?? flag.type;

  return (
    <article
      className="flex items-start gap-6 bg-white px-7 py-6"
      style={{ border: "1px solid #E4E4E4" }}
    >
      {/* Left rail: icon box + state label. */}
      <div className="flex w-[84px] shrink-0 flex-col gap-2.5 pt-0.5">
        <Signal status={flag.status} />
        <div
          className="text-base font-semibold leading-tight"
          style={{ color: meta.labelColor }}
        >
          {meta.label}
        </div>
      </div>

      {/* Body */}
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <h3 className="text-xl font-semibold tracking-tight text-ink">{typeLabel}</h3>
          <div className="flex flex-wrap items-center gap-2">
            {flag.status === "fired" && flag.severity && (
              <span
                className="px-2.5 py-[3px] text-base font-medium"
                style={{ color: "#8A5A12", border: "1px solid #D9B877" }}
              >
                {SEVERITY_WORD[flag.severity] ?? "Notable"}
              </span>
            )}
            {flag.source === "llm" && (
              <span
                className="px-2.5 py-[3px] text-base font-medium text-muted"
                style={{ border: "1px solid #D6D6D6" }}
              >
                AI second opinion
              </span>
            )}
          </div>
        </div>

        <p className="mt-2.5 max-w-[64ch] text-base leading-relaxed text-ink/80">
          {flag.plain_explanation}
        </p>

        {/* A check that could not run says why, in the card itself. */}
        {flag.status === "not_assessed" && flag.evidence && (
          <p className="mt-2 max-w-[64ch] text-base leading-relaxed text-ink/70">
            <span className="font-semibold text-ink">Why: </span>
            {flag.evidence}
          </p>
        )}

        {/* Moments this finding cites: each jumps the player there. */}
        {onSeek && flag.timestamps?.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="font-mono text-base uppercase tracking-[0.14em] text-muted">
              {flag.timestamps.length === 1 ? "Moment" : "Moments"}
            </span>
            {flag.timestamps.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => onSeek(t)}
                className="min-h-[44px] px-3 py-2 font-mono text-base text-ink hover:bg-ink hover:text-white"
                style={{ border: "1px solid #161616" }}
                title={`Play from ${clock(t)}`}
              >
                ▶ {clock(t)}
              </button>
            ))}
          </div>
        )}

        {flag.what_to_check && (
          <div className="mt-4">
            <div className="font-mono text-base uppercase tracking-[0.14em] text-muted">
              {CHECK_HEADING[flag.status] ?? "What to check"}
            </div>
            <p className="mt-1 max-w-[64ch] text-base leading-relaxed text-ink/75">
              {flag.what_to_check}
            </p>
          </div>
        )}

        {flag.evidence && (
          <details className="group mt-4">
            <summary className="flex min-h-[44px] items-center gap-1.5 py-2 font-mono text-base uppercase tracking-[0.12em] text-muted hover:text-ink">
              <span className="chev inline-block transition-transform">▸</span>
              Supporting evidence
            </summary>
            <p
              className="mt-2 max-w-[68ch] whitespace-pre-wrap break-words bg-panel px-3 py-2 font-mono text-base leading-relaxed text-ink/75"
              style={{ border: "1px solid #E4E4E4" }}
            >
              {flag.evidence}
            </p>
          </details>
        )}
      </div>
    </article>
  );
}
