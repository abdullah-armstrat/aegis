// One finding, rendered in the dossier style and bound to the real backend Flag:
//   { type, status, severity, evidence, plain_explanation, what_to_check, source }
// Spine: explain, don't verdict. The three states read by shape (Signal) before colour, and
// "couldn't check" (not_assessed) is visually distinct from "clear" — never conflated (ADR-009).

import Signal, { STATUS_META } from "./Signal";

// Human labels for the flag types the backend actually emits today.
const TYPE_LABELS = {
  caption_content_mismatch: "Caption ↔ image match",
  recycled_context: "Recycled context",
  emotional_framing: "Emotional framing",
  audio_visual_mismatch: "Audio ↔ visual match",
  ai_generation_hint: "AI-generation hint",
};

// Map the backend severity enum (info|low|medium|high) to dossier weight words.
const SEVERITY_WORD = { high: "Notable", medium: "Notable", low: "Minor", info: "Minor" };

// Per-status card treatment.
const CARD = {
  fired: "border-flag/40 bg-flagbg",
  clear: "border-clear/30 bg-clearbg",
  not_assessed: "border-dashed border-line bg-panel",
};

// "What was attempted" reads better than "What to check" when nothing could be assessed,
// but our backend supplies a single what_to_check string; we relabel the heading by status.
const CHECK_HEADING = {
  fired: "What to check",
  clear: "What to check",
  not_assessed: "What to check yourself",
};

export default function FlagCard({ flag }) {
  const meta = STATUS_META[flag.status] ?? STATUS_META.not_assessed;
  const typeLabel = TYPE_LABELS[flag.type] ?? flag.type;
  const cardClass = CARD[flag.status] ?? CARD.not_assessed;

  return (
    <article className={`border ${cardClass} px-5 py-4`}>
      {/* Status line — shape + label, the dossier's signalling row. */}
      <div className="flex items-center gap-2">
        <Signal status={flag.status} />
        <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
          {meta.label}
        </span>
        {flag.status === "fired" && flag.severity && (
          <span className="ml-1 font-mono text-[11px] uppercase tracking-[0.14em] text-flag">
            · {SEVERITY_WORD[flag.severity] ?? "Notable"}
          </span>
        )}
      </div>

      {/* Title + optional AI-second-opinion tag. */}
      <div className="mt-2 flex items-baseline justify-between gap-3">
        <h3 className="text-lg font-semibold tracking-tight text-ink">{typeLabel}</h3>
        {flag.source === "llm" && (
          <span className="shrink-0 border border-line px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
            AI second opinion
          </span>
        )}
      </div>

      {/* Plain explanation. */}
      <p className="mt-1.5 text-[15px] leading-relaxed text-ink/85">{flag.plain_explanation}</p>

      {/* "What to check" — the durable, teach-the-user value. */}
      {flag.what_to_check && (
        <div className="mt-3 border-l-2 border-ink/15 pl-3">
          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
            {CHECK_HEADING[flag.status] ?? "What to check"}
          </div>
          <p className="mt-0.5 text-sm leading-relaxed text-ink/80">{flag.what_to_check}</p>
        </div>
      )}

      {/* Supporting evidence — the actual backend evidence string, on demand. */}
      {flag.evidence && (
        <details className="mt-3 group">
          <summary className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.12em] text-muted hover:text-ink">
            <span className="chev inline-block transition-transform">▸</span>
            Supporting evidence
          </summary>
          <p className="mt-2 whitespace-pre-wrap break-words border border-line bg-white px-3 py-2 font-mono text-[12px] leading-relaxed text-ink/75">
            {flag.evidence}
          </p>
        </details>
      )}
    </article>
  );
}
