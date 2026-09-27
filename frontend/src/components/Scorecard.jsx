// The "Audit result" screen: a neutral, count-based summary (not a verdict), the post under
// review, then one FlagCard per check. Findings are sorted fired → not_assessed → clear so the
// user sees what to check first, while coverage (what couldn't be checked) stays visible.

import { useRef } from "react";
import FlagCard, { clock } from "./FlagCard";

const STATUS_ORDER = { fired: 0, not_assessed: 1, clear: 2 };

// "two", "one"… for the small count sentence; falls back to the numeral past nine.
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const word = (n) => WORDS[n] ?? String(n);

function countSentence(flags) {
  const fired = flags.filter((f) => f.status === "fired").length;
  const na = flags.filter((f) => f.status === "not_assessed").length;
  const clear = flags.filter((f) => f.status === "clear").length;
  const parts = [];
  if (fired) parts.push(`${word(fired)} flag${fired === 1 ? "" : "s"} raised`);
  if (na) parts.push(`${word(na)} couldn’t be checked`);
  if (clear) parts.push(`${word(clear)} came back clear`);
  const checks = `${word(flags.length).replace(/^\w/, (c) => c.toUpperCase())} check${
    flags.length === 1 ? "" : "s"
  } run.`;
  // Each part is a new sentence after "N checks run.", so it starts with a capital.
  const rest = parts.join(", ").replace(/^\w/, (c) => c.toUpperCase());
  return parts.length ? `${checks} ${rest}.` : checks;
}

// The video under review: the player, and a strip of the keyframes Aegis looked at, each with
// its time. Clicking a keyframe (or a time on a finding) moves the player to that moment.
function VideoReview({ src, keyframes, duration, playerRef, onSeek }) {
  return (
    <div className="mt-5 border border-line bg-panel p-4">
      <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
        Video under review{duration ? ` · ${clock(duration)}` : ""}
      </div>
      {src && (
        <video ref={playerRef} src={src} controls className="mt-3 max-h-[360px] w-full bg-black" />
      )}
      {keyframes?.length > 0 && (
        <div className="mt-3">
          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
            Keyframes checked ({keyframes.length})
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {keyframes.map((k) => (
              <button
                key={k.timestamp}
                type="button"
                onClick={() => onSeek(k.timestamp)}
                className="shrink-0 border border-line bg-white p-1 text-left hover:border-ink"
                title={`Go to ${clock(k.timestamp)}`}
              >
                {k.thumbnail && (
                  <img src={k.thumbnail} alt={`frame at ${clock(k.timestamp)}`} className="h-16 w-auto" />
                )}
                <span className="mt-1 block font-mono text-[11px] text-ink">{clock(k.timestamp)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Scorecard({ scorecard, submission }) {
  const playerRef = useRef(null);
  if (!scorecard) return null;
  const isVideo = scorecard.modality === "video";

  function seek(seconds) {
    const player = playerRef.current;
    if (!player) return;
    player.currentTime = seconds;
    player.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const flags = [...(scorecard.flags ?? [])].sort(
    (a, b) => (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9)
  );

  return (
    <section>
      {/* Result header. */}
      <div className="flex items-baseline justify-between border-b border-ink pb-2">
        <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-ink">Audit result</h2>
        {scorecard.source_ref && (
          <span className="font-mono text-[11px] text-muted">{scorecard.source_ref}</span>
        )}
      </div>

      {/* Count summary + the non-verdict reminder (the spine). */}
      <p className="mt-4 text-xl font-medium leading-snug tracking-tight text-ink">
        {countSentence(flags)}
      </p>
      <p className="mt-2 max-w-prose text-sm leading-relaxed text-muted">
        {scorecard.summary ? `${scorecard.summary} ` : ""}
        Aegis doesn’t decide what is true. It surfaces specific, checkable findings and shows you
        how to weigh them. The judgement stays with you.
      </p>

      {isVideo && (
        <VideoReview
          src={submission?.preview}
          keyframes={scorecard.keyframes}
          duration={scorecard.duration_s}
          playerRef={playerRef}
          onSeek={seek}
        />
      )}

      {/* The post under review. */}
      {submission && (
        <div className="mt-5 border border-line bg-panel p-4">
          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
            Under review
          </div>
          <div className="mt-3 flex gap-4">
            {submission.preview && !isVideo && (
              <img
                src={submission.preview}
                alt="submitted"
                className="h-20 w-20 shrink-0 border border-line object-cover"
              />
            )}
            <div className="min-w-0">
              {submission.caption ? (
                <p className="text-sm italic leading-relaxed text-ink/80">
                  “{submission.caption}”
                </p>
              ) : (
                <p className="text-sm text-muted">No caption provided.</p>
              )}
              <p className="mt-2 font-mono text-[11px] text-muted">
                {submission.filename ? `${submission.filename} · ` : ""}
                {submission.postedDate ? `posted ${submission.postedDate} · ` : ""}
                {submission.caption
                  ? `caption ${submission.caption.trim().split(/\s+/).length} words`
                  : "no caption"}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Findings. */}
      <div className="mt-6">
        <div className="flex items-baseline justify-between border-b border-line pb-1.5">
          <h3 className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
            Findings ({flags.length})
          </h3>
        </div>
        <div className="mt-4 space-y-3">
          {flags.map((flag, i) => (
            <FlagCard
              key={`${flag.type}-${flag.source}-${i}`}
              flag={flag}
              onSeek={isVideo ? seek : undefined}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
