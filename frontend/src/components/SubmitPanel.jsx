// Submit screen: two-column dossier layout EXACTLY per the design: image or video (solid grey
// dropzone, left) beside caption (textarea, right), then an action row with the ink "Run
// audit" button and a right-aligned privacy note. Reports the submission up so the result
// screen can show what was reviewed.

import { useRef, useState } from "react";
import { isVideo } from "../api";

export default function SubmitPanel({ onAnalyze, loading, webSearchAvailable = false }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [postedDate, setPostedDate] = useState("");
  const [searchWeb, setSearchWeb] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function handleFile(selected) {
    if (!selected || !(selected.type?.startsWith("image/") || isVideo(selected))) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  const video = isVideo(file);

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (file && !loading)
      onAnalyze(file, caption, {
        preview,
        filename: file.name,
        postedDate,
        video,
        searchWeb: searchWeb && !video,
      });
  }

  return (
    <form onSubmit={handleSubmit} className="pb-20 pt-2">
      <h1 className="text-4xl font-semibold leading-[1.1] tracking-tight text-ink">
        Submit a post for review
      </h1>
      <p className="mt-4 max-w-[540px] text-base leading-relaxed text-[#595959]">
        Aegis reads the image or video and its caption together, then reports what it found, finding
        by finding. It will not tell you whether the post is true; it shows you what is worth checking.
      </p>

      {/* Two-column grid: image | caption */}
      <div className="mt-11 grid grid-cols-1 items-start gap-8 md:grid-cols-2">
        {/* Image */}
        <div>
          <div className="mb-3 flex items-baseline justify-between">
            <span className="text-base font-semibold text-ink">The image or video</span>
            <span className="text-base text-muted">Required</span>
          </div>
          <button
            type="button"
            aria-label={file ? `Chosen file: ${file.name}. Choose a different image or video` : "Choose an image or video"}
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
            {preview && video ? (
              // "#t=0.1" and preload make the browser draw a frame instead of an empty box.
              <video
                src={`${preview}#t=0.1`}
                preload="metadata"
                muted
                playsInline
                className="max-h-full max-w-full object-contain"
              />
            ) : preview ? (
              <img src={preview} alt="preview" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-3 px-6 text-center">
                {/* Image-placeholder icon (matches the design). */}
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="#9A9A9A" strokeWidth="1.5" />
                  <circle cx="8.5" cy="8.5" r="1.8" fill="#9A9A9A" />
                  <path d="M21 15l-5-5L5 21" stroke="#9A9A9A" strokeWidth="1.5" />
                </svg>
                <span className="text-base text-muted">
                  Drag an image or video here, or{" "}
                  <span className="text-ink underline">click to browse</span>
                </span>
                <span className="text-base text-muted">Videos: mp4, mov or webm, up to 60 seconds</span>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*,video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </button>
          {file && <p className="mt-2 font-mono text-base text-muted">{file.name}</p>}
        </div>

        {/* Caption */}
        <div>
          <div className="mb-3 flex items-baseline justify-between">
            <span className="text-base font-semibold text-ink">The caption</span>
            <span className="text-base text-muted">As posted</span>
          </div>
          <textarea
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            placeholder="Paste the full caption here. Keep the original wording, hashtags, links and emoji."
            className="h-[360px] w-full resize-y border bg-white px-[19px] py-[17px] text-base leading-relaxed text-ink placeholder:text-muted focus:border-ink focus:outline-none"
            style={{ borderColor: "#D6D6D6" }}
          />
        </div>
      </div>

      {/* Optional posting date: lets recycled context compare earlier appearances with it. */}
      <div className="mt-8 flex flex-wrap items-end gap-x-6 gap-y-2">
        <label className="flex flex-col gap-2">
          <span className="flex items-baseline gap-3">
            <span className="text-base font-semibold text-ink">Date posted</span>
            <span className="text-base text-muted">Optional</span>
          </span>
          <input
            type="date"
            value={postedDate}
            max={new Date().toISOString().slice(0, 10)}
            onChange={(e) => setPostedDate(e.target.value)}
            className="min-h-[44px] border bg-white px-3 py-2 text-base text-ink focus:border-ink focus:outline-none"
            style={{ borderColor: "#D6D6D6" }}
          />
        </label>
        <p className="max-w-[420px] pb-2 text-base leading-relaxed text-muted">
          If you know when the post was published, add it. Aegis can then say whether the image, or
          a frame of the video, was online before that date, rather than only whether it has
          appeared elsewhere.
        </p>
      </div>

      {/* Optional web search: offered only when the server has a key, and only for images. */}
      {webSearchAvailable && file && !video && (
        <label className="mt-6 flex max-w-[640px] items-start gap-3">
          <input
            type="checkbox"
            checked={searchWeb}
            onChange={(e) => setSearchWeb(e.target.checked)}
            className="mt-0.5 h-7 w-7 shrink-0 accent-ink"
          />
          <span className="text-base leading-relaxed text-ink">
            <span className="font-semibold">Also search the web for this image.</span>{" "}
            <span className="text-muted">
              The image will be sent to Google (Cloud Vision) to find pages that show it. Without
              this, Aegis only checks its own offline index.
            </span>
          </span>
        </label>
      )}

      {/* Action row */}
      <div
        className="mt-9 flex flex-wrap items-center justify-between gap-6 pt-6"
        style={{ borderTop: "1px solid #E4E4E4" }}
      >
        <button
          type="submit"
          disabled={!file || loading}
          className="inline-flex items-center gap-2.5 px-7 py-3.5 text-base font-semibold tracking-tight text-white transition-colors"
          style={{
            background: !file || loading ? "#8C8C8C" : "#161616",
            cursor: !file || loading ? "not-allowed" : "pointer",
          }}
          title={!file ? "Choose an image or video first" : "Run the audit"}
        >
          {loading ? (video ? "Running audit… a video takes up to a minute" : "Running audit…") : "Run audit"}
          <span className="text-base leading-none">→</span>
        </button>
        <p className="max-w-[280px] text-right text-base leading-relaxed text-muted">
          Nothing is published. Your file and caption stay in this session.
        </p>
      </div>
    </form>
  );
}
