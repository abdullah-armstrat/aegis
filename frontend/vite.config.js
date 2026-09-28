import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server runs on :5173 and talks to the FastAPI backend on :8000
// (CORS is set up in the backend; the base URL comes from VITE_API_BASE).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
