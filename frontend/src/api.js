// Thin client for the Aegis FastAPI backend.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function getHealth() {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) throw new Error(`health ${resp.status}`);
  return resp.json();
}

// analyze(file, caption, postedDate) -> Scorecard JSON from POST /analyze.
// postedDate is optional ("YYYY-MM-DD"); recycled context compares earlier appearances with it.
export async function analyze(file, caption, postedDate) {
  const form = new FormData();
  form.append("image", file);
  form.append("caption", caption ?? "");
  if (postedDate) form.append("posted_date", postedDate);

  const resp = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
  if (!resp.ok) {
    let detail = `analyze ${resp.status}`;
    try {
      const body = await resp.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status-based message */
    }
    throw new Error(detail);
  }
  return resp.json();
}
