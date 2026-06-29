// Submit screen: two-column dossier layout EXACTLY per the design: image (solid grey
// dropzone, left) beside caption (textarea, right), then an action row with the ink "Run
// audit" button and a right-aligned privacy note. Reports the submission up so the result
// screen can show what was reviewed.

import { useRef, useState } from "react";

export default function SubmitPanel({ onAnalyze, loading }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function handleFile(selected) {
    if (!selected || !selected.type?.startsWith("image/")) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (file && !loading) onAnalyze(file, caption, { preview, filename: file.name });
  }

  return (
    <form onSubmit={handleSubmit} className="pb-20 pt-2">
      <h1 className="text-4xl font-semibold leading-[1.1] tracking-tight text-ink">
        Submit a post for review
      </h1>
      <p className="mt-4 max-w-[540px] text-base leading-relaxed text-[#595959]">
        Aegis reads the image and its caption together, then reports what it found, finding by
        finding. It will not tell you whether the post is true; it shows you what is worth checking.
      </p>

      {/* Two-column grid: image | caption */}
      <div className="mt-11 grid grid-cols-1 items-start gap-8 md:grid-cols-2">
        {/* Image */}
        <div>
          <div className="mb-3 flex items-baseline justify-between">
            <span className="text-[13px] font-semibold text-ink">The image</span>
            <span className="text-xs text-muted">Required</span>
          </div>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className="flex h-[360px] w-full items-center justify-center overflow-hidden border border-dashed bg-fill transition-colors"
            style={{ borderColor: dragging ? "#161616" : "#C9C9C9" }}
          >
            {preview ? (
              <img src={preview} alt="preview" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-3 px-6 text-center">
                {/* Image-placeholder icon (matches the design). */}
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="#9A9A9A" strokeWidth="1.5" />
                  <circle cx="8.5" cy="8.5" r="1.8" fill="#9A9A9A" />
                  <path d="M21 15l-5-5L5 21" stroke="#9A9A9A" strokeWidth="1.5" />
                </svg>
                <span className="text-sm text-muted">
                  Drag an image here, or <span className="text-ink underline">click to browse</span>
                </span>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </button>
          {file && <p className="mt-2 font-mono text-[11px] text-muted">{file.name}</p>}
        </div>

        {/* Caption */}
        <div>
          <div className="mb-3 flex items-baseline justify-between">
            <span className="text-[13px] font-semibold text-ink">The caption</span>
            <span className="text-xs text-muted">As posted</span>
          </div>
          <textarea
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            placeholder="Paste the full caption here. Keep the original wording, hashtags, links and emoji."
            className="h-[360px] w-full resize-y border bg-white px-[19px] py-[17px] text-[15px] leading-relaxed text-ink placeholder:text-muted focus:border-ink focus:outline-none"
            style={{ borderColor: "#D6D6D6" }}
          />
        </div>
      </div>

      {/* Action row */}
      <div
        className="mt-9 flex flex-wrap items-center justify-between gap-6 pt-6"
        style={{ borderTop: "1px solid #E4E4E4" }}
      >
        <button
          type="submit"
          disabled={!file || loading}
          className="inline-flex items-center gap-2.5 px-7 py-3.5 text-[15px] font-semibold tracking-tight text-white transition-colors"
          style={{
            background: !file || loading ? "#8C8C8C" : "#161616",
            cursor: !file || loading ? "not-allowed" : "pointer",
          }}
          title={!file ? "Choose an image first" : "Run the audit"}
        >
          {loading ? "Running audit…" : "Run audit"}
          <span className="text-base leading-none">→</span>
        </button>
        <p className="max-w-[280px] text-right text-[12.5px] leading-relaxed text-muted">
          Nothing is published. Your image and caption stay in this session.
        </p>
      </div>
    </form>
  );
}
