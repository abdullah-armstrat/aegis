// Submit screen, dossier style: choose an image (with preview) + paste the caption, then run
// the audit. Reports the submission back up so the result screen can show what was reviewed.

import { useRef, useState } from "react";

export default function SubmitPanel({ onAnalyze, loading }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const inputRef = useRef(null);

  function handleFile(selected) {
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (file) onAnalyze(file, caption, { preview, filename: file.name });
  }

  return (
    <form onSubmit={handleSubmit}>
      <h2 className="text-2xl font-semibold tracking-tight text-ink">Submit a post for review</h2>
      <p className="mt-2 max-w-prose text-[15px] leading-relaxed text-muted">
        Aegis reads the image and its caption together, then reports what it found, finding by
        finding. It will not tell you whether the post is true; it shows you what is worth checking.
      </p>

      {/* Image */}
      <div className="mt-7 flex items-baseline justify-between">
        <label className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink">
          The image
        </label>
        <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
          Required
        </span>
      </div>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-2 flex w-full flex-col items-center justify-center border border-dashed border-line bg-panel px-4 py-10 text-center transition-colors hover:border-ink/40"
      >
        {preview ? (
          <img src={preview} alt="preview" className="max-h-56 border border-line object-contain" />
        ) : (
          <span className="text-sm text-muted">Click to choose an image to audit</span>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </button>
      {file && (
        <p className="mt-1.5 font-mono text-[11px] text-muted">{file.name}</p>
      )}

      {/* Caption */}
      <div className="mt-6 flex items-baseline justify-between">
        <label htmlFor="caption" className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink">
          The caption
        </label>
        <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
          As posted
        </span>
      </div>
      <textarea
        id="caption"
        value={caption}
        onChange={(e) => setCaption(e.target.value)}
        rows={4}
        placeholder="Paste the caption exactly as it appears with the post…"
        className="mt-2 w-full resize-y border border-line bg-white p-3 text-[15px] leading-relaxed text-ink placeholder:text-muted focus:border-ink focus:outline-none"
      />

      <button
        type="submit"
        disabled={!file || loading}
        className="mt-6 inline-flex items-center gap-2 bg-ink px-6 py-3 text-sm font-semibold tracking-tight text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? "Running audit…" : "Run audit →"}
      </button>

      <p className="mt-4 font-mono text-[11px] leading-relaxed text-muted">
        Nothing is published. Your image and caption stay in this session.
      </p>
    </form>
  );
}
