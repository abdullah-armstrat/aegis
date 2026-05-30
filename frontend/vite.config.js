import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev server runs on :5173 and talks to the FastAPI backend on :8000
// (CORS is configured backend-side; the base URL is set via VITE_API_BASE).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
