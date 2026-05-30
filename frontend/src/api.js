// Thin client for the Aegis FastAPI backend.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function getHealth() {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) throw new Error(`health ${resp.status}`);
  return resp.json();
}

// analyze(image, caption) -> Scorecard JSON. Wired in Week 2 once /analyze exists.
export async function analyze(/* file, caption */) {
  throw new Error("Not implemented yet — arrives with the /analyze endpoint in Week 2.");
}
