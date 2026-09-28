// Interface review: the real backend and the Vite dev server in Google Chrome, as in the
// browser tests. Run with `npm run review:interface`; set ROUND=B for round B.
import { defineConfig } from "@playwright/test";

const python =
  process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";

export default defineConfig({
  testDir: "./review",
  timeout: 1_800_000,
  expect: { timeout: 60_000 },
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
      // A fake key shows the web-search option and its notice, which the review covers; nothing
      // ticks it, and the real key is never given to this server.
      env: { AEGIS_USE_LLM: "false", GOOGLE_VISION_API_KEY: "review-not-a-real-key" },
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://localhost:5173",
      timeout: 60_000,
      reuseExistingServer: false,
    },
  ],
});
