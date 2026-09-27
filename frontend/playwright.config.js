// End-to-end tests: the real backend and the Vite dev server, driven in Google Chrome.
// Run with `npm run test:e2e`. The backend runs from backend/.venv with the network to
// Hugging Face disabled and the LLM off, as in the default install.
import { defineConfig } from "@playwright/test";

const python =
  process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";

export default defineConfig({
  testDir: "./e2e",
  timeout: 240_000,
  expect: { timeout: 180_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5173",
    channel: "chrome",
    headless: true,
  },
  webServer: [
    {
      command: `${python} -m uvicorn app.main:app --port 8000`,
      cwd: "../backend",
      url: "http://localhost:8000/health",
      timeout: 120_000,
      reuseExistingServer: false,
      env: { HF_HUB_OFFLINE: "1", AEGIS_USE_LLM: "false" },
    },
    {
      command: "npm run dev -- --port 5173 --strictPort",
      url: "http://localhost:5173",
      timeout: 60_000,
      reuseExistingServer: false,
    },
  ],
});
