// A single flag: its type, status, plain explanation, and "what to check" prompt.
// Reflects the project's spine — explain, don't verdict. The status (fired / clear /
// not_assessed) is shown honestly so "couldn't check" never looks like "all clear".

const STATUS_STYLES = {
  fired: { label: "Flag raised", box: "border-amber-300 bg-amber-50", dot: "bg-amber-500" },
  clear: { label: "Checked — clear", box: "border-emerald-200 bg-emerald-50", dot: "bg-emerald-500" },
  not_assessed: { label: "Could not check", box: "border-slate-200 bg-slate-50", dot: "bg-slate-400" },
};

const TYPE_LABELS = {
  caption_content_mismatch: "Caption ↔ image match",
  audio_visual_mismatch: "Audio ↔ visual match",
  recycled_context: "Recycled context",
  emotional_framing: "Emotional framing",
  ai_generation_hint: "AI-generation hint",
};

const SEVERITY_BADGE = {
  high: "bg-red-100 text-red-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-sky-100 text-sky-700",
  info: "bg-slate-100 text-slate-600",
};

export default function FlagCard({ flag }) {
  const status = STATUS_STYLES[flag.status] ?? STATUS_STYLES.not_assessed;
  const typeLabel = TYPE_LABELS[flag.type] ?? flag.type;

  return (
    <div className={`rounded-xl border p-4 ${status.box}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${status.dot}`} />
          <h3 className="font-semibold text-slate-800">{typeLabel}</h3>
        </div>
        <div className="flex items-center gap-2">
          {flag.status === "fired" && (
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                SEVERITY_BADGE[flag.severity] ?? SEVERITY_BADGE.info
              }`}
            >
              {flag.severity}
            </span>
          )}
          <span className="text-xs uppercase tracking-wide text-slate-500">{status.label}</span>
        </div>
      </div>

      <p className="mt-2 text-sm text-slate-700">{flag.plain_explanation}</p>

      {flag.what_to_check && (
        <p className="mt-2 text-sm text-slate-600">
          <span className="font-medium text-slate-700">What to check: </span>
          {flag.what_to_check}
        </p>
      )}

      {flag.source === "llm" && (
        <p className="mt-2 text-xs italic text-slate-400">
          From the language-model reasoner (a second opinion over the extracted text).
        </p>
      )}
    </div>
  );
}
