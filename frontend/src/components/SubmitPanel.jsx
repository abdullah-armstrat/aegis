// Upload an image + caption and submit for analysis. Shows a local preview, basic validation,
// and a loading state while /analyze runs.

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
    if (file) onAnalyze(file, caption);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
    >
      <label className="block text-sm font-medium text-slate-700">Image</label>
      <div
        onClick={() => inputRef.current?.click()}
        className="mt-2 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 px-4 py-8 text-center hover:border-slate-400"
      >
        {preview ? (
          <img src={preview} alt="preview" className="max-h-48 rounded-md object-contain" />
        ) : (
          <span className="text-sm text-slate-500">Click to choose an image to audit</span>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      <label htmlFor="caption" className="mt-4 block text-sm font-medium text-slate-700">
        Caption
      </label>
      <textarea
        id="caption"
        value={caption}
        onChange={(e) => setCaption(e.target.value)}
        rows={3}
        placeholder="Paste the caption that accompanies this image…"
        className="mt-2 w-full rounded-lg border border-slate-300 p-2 text-sm focus:border-slate-500 focus:outline-none"
      />

      <button
        type="submit"
        disabled={!file || loading}
        className="mt-4 w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {loading ? "Analysing…" : "Analyse"}
      </button>
    </form>
  );
}
