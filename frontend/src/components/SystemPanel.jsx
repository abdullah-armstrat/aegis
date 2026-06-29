// "System" page: the design's style/about page, made honest: the dossier palette, the type
// system, the three-state signalling explained, and Aegis's explicit non-goals (no fakery /
// AI-generation detection). Static; no backend calls.

import Signal from "./Signal";

const SWATCHES = [
  { name: "White", hex: "#FFFFFF" },
  { name: "Panel", hex: "#FAFAFA" },
  { name: "Fill", hex: "#F0F0F0" },
  { name: "Line", hex: "#D6D6D6" },
  { name: "Muted", hex: "#8C8C8C" },
  { name: "Ink", hex: "#161616" },
];

const SIGNALS = [
  { status: "fired", text: "A flag set in amber carries the most weight. Something is worth checking." },
  { status: "not_assessed", text: "A broken ring on a dashed card. The check couldn’t run. It is never a clear." },
  { status: "clear", text: "A closed ring with a tick, set in a quiet green. Looked, found nothing." },
];

function Heading({ children }) {
  return (
    <h2 className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted">{children}</h2>
  );
}

export default function SystemPanel() {
  return (
    <div className="py-14">
      <h1 className="text-3xl font-semibold tracking-tight text-ink">The system</h1>
      <p className="mt-3 max-w-[560px] text-[15px] leading-relaxed text-[#595959]">
        A black-and-white dossier palette built on white, with grey tones for containers, and a
        three-part signalling system distinguished by shape and weight before any colour.
      </p>

      {/* Greyscale */}
      <div className="mt-10">
        <Heading>Greyscale</Heading>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-6">
          {SWATCHES.map((s) => (
            <div key={s.name} style={{ border: "1px solid #E4E4E4" }}>
              <div className="h-16 w-full" style={{ background: s.hex }} />
              <div className="px-2 py-1.5">
                <div className="text-[12px] font-semibold text-ink">{s.name}</div>
                <div className="font-mono text-[10px] text-muted">{s.hex}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Type */}
      <div className="mt-10">
        <Heading>Type</Heading>
        <div className="mt-4 space-y-3" style={{ border: "1px solid #E4E4E4" }}>
          <div className="px-4 py-3">
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              Archivo · headings
            </div>
            <div className="mt-1 text-2xl font-semibold tracking-tight text-ink">
              Recycled context
            </div>
          </div>
          <div className="px-4 py-3" style={{ borderTop: "1px solid #E4E4E4" }}>
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              Archivo · body
            </div>
            <div className="mt-1 text-[15px] text-ink/85">
              Aegis surfaces specific, checkable findings. The judgement stays with you.
            </div>
          </div>
          <div className="px-4 py-3" style={{ borderTop: "1px solid #E4E4E4" }}>
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              IBM Plex Mono · metadata
            </div>
            <div className="mt-1 font-mono text-[13px] text-ink/75">earliest copy · 12 apr 2024</div>
          </div>
        </div>
      </div>

      {/* Signalling */}
      <div className="mt-10">
        <Heading>Signalling: shape, weight and colour</Heading>
        <div className="mt-4 space-y-3">
          {SIGNALS.map((s) => (
            <div
              key={s.status}
              className="flex items-center gap-4 bg-white px-5 py-4"
              style={{ border: "1px solid #E4E4E4" }}
            >
              <Signal status={s.status} />
              <p className="text-sm leading-relaxed text-ink/80">{s.text}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Non-goals, the honest part. */}
      <div className="mt-10">
        <Heading>What Aegis does not do</Heading>
        <p className="mt-3 max-w-[560px] text-[15px] leading-relaxed text-[#595959]">
          Aegis does not decide whether a post is true or false, and does not claim to detect
          AI-generated or digitally-manipulated images; those are unsolved, unreliable tasks. It
          reports only what its checks can reliably surface: cross-modal inconsistencies, recycled
          context, and emotional framing. Findings are leads for you to verify, not rulings.
        </p>
      </div>
    </div>
  );
}
