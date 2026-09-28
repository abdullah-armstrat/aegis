// The report's screenshots: the real backend in its default mode (offline, no web-search key, the
// language model off) and the Vite dev server, in Google Chrome. Run with `npm run screenshots`.
import { defineConfig } from "@playwright/test";

const python =
  process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";

export default defineConfig({
  testDir: "./screenshots",
  timeout: 600_000,
  expect: { timeout: 240_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5173",
    channel: "chrome",
    headless: true,
    viewport: { width: 1366, height: 768 },
  },
  webServer: [
    {
      command: `${python} -m uvicorn app.main:app --port 8000`,
      cwd: "../backend",
      url: "http://localhost:8000/health",
      timeout: 120_000,
      reuseExistingServer: false,
      env: { AEGIS_USE_LLM: "false", GOOGLE_VISION_API_KEY: "" },
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://localhost:5173",
      timeout: 60_000,
      reuseExistingServer: false,
    },
  ],
});
