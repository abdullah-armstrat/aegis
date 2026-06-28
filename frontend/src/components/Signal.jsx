// The three-state signal, distinguished by SHAPE and WEIGHT before colour (so it reads for
// colour-blind users too) — the core of the dossier design and of ADR-009:
//   fired        → a solid amber marker (a raised flag): something to check.
//   not_assessed → a broken ring on a dashed treatment: the check could not run. Never a clear.
//   clear        → a closed ring with a tick, quiet green: looked, found nothing.
//
// `flag.status` from the backend is one of: "fired" | "clear" | "not_assessed".

export const STATUS_META = {
  fired: { label: "Flag raised" },
  not_assessed: { label: "Couldn’t check" },
  clear: { label: "Checked, clear" },
};

export default function Signal({ status, size = 18 }) {
  const s = size;
  const c = s / 2;
  const r = s / 2 - 2;

  if (status === "fired") {
    // A small flag on a pole — the heaviest mark.
    return (
      <svg width={s} height={s} viewBox="0 0 18 18" aria-hidden="true" role="img">
        <rect x="4" y="2" width="1.6" height="14" rx="0.8" fill="#B7791F" />
        <path d="M5.6 3 L14 5.2 L5.6 7.4 Z" fill="#B7791F" />
      </svg>
    );
  }

  if (status === "clear") {
    // Closed ring + tick, quiet green.
    return (
      <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} aria-hidden="true" role="img">
        <circle cx={c} cy={c} r={r} fill="none" stroke="#3F7A52" strokeWidth="1.4" />
        <path
          d={`M${c - 3} ${c} L${c - 0.8} ${c + 2.2} L${c + 3.2} ${c - 2.4}`}
          fill="none"
          stroke="#3F7A52"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  // not_assessed (default): a BROKEN ring — deliberately incomplete, muted grey.
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} aria-hidden="true" role="img">
      <circle
        cx={c}
        cy={c}
        r={r}
        fill="none"
        stroke="#8C8C8C"
        strokeWidth="1.4"
        strokeDasharray="3 2.4"
      />
    </svg>
  );
}
