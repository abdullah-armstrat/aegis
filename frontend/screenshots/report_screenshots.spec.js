// The report's screenshots, saved to results/figures/. Every picture and sound in them is made
// here or by the project: the three illustrative images drawn by build_image_index.py, a video
// made from them with ffmpeg, and narration spoken by the Windows voice (on other systems the
// video has no speech). No third-party or personal media appears.
import { test, expect } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ROOT = resolve("..");
const OUT = join(ROOT, "results", "figures");
const ILLUSTRATIVE = join(ROOT, "data", "illustrative");
const work = mkdtempSync(join(tmpdir(), "aegis-screens-"));
mkdirSync(OUT, { recursive: true });

const IMAGE = join(ILLUSTRATIVE, "flood_illustrative.png");
const IMAGE_CAPTION = "BREAKING: the city centre is under water TODAY. Share this now before they delete it!!";
const POSTED = "2026-09-27";
const VIDEO_CAPTION = "Floods hit the town, then the evening turns calm";
// Each scene lasts 7 s; each line starts half a second into its scene. The third does not fit.
const SCENES = ["flood_illustrative.png", "sunset_illustrative.png", "cat_illustrative.png"];
const LINES = [
  "Brown floodwater covers the ground around the buildings.",
  "The sun sets over a calm, dark sea.",
  "A crowd of people marches through a busy city square.",
];

function ffmpeg(args) {
  execFileSync("ffmpeg", ["-v", "error", "-y", ...args]);
}

function speak(text, wav) {
  const script =
    "Add-Type -AssemblyName System.Speech; " +
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; " +
    `$s.SetOutputToWaveFile('${wav}'); $s.Speak('${text.replace(/'/g, "''")}'); $s.Dispose()`;
  execFileSync("powershell", ["-NoProfile", "-Command", script]);
}

function makeVideo() {
  const inputs = SCENES.flatMap((f) => ["-loop", "1", "-t", "7", "-framerate", "10", "-i", join(ILLUSTRATIVE, f)]);
  const scale = SCENES.map((_, i) => `[${i}:v]scale=640:426,setsar=1[v${i}]`).join(";");
  const video = `${scale};[v0][v1][v2]concat=n=3:v=1:a=0[v]`;
  const out = join(work, "three-scenes-narrated.mp4");
  if (process.platform !== "win32") {
    ffmpeg([...inputs, "-filter_complex", video, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", out]);
    return out;
  }
  const wavs = LINES.map((line, i) => {
    const wav = join(work, `line${i}.wav`);
    speak(line, wav);
    return wav;
  });
  const audio = wavs.map((_, i) => `[${i + 3}:a]adelay=${i * 7000 + 500}|${i * 7000 + 500}[a${i}]`).join(";") +
    ";[a0][a1][a2]amix=inputs=3:normalize=0[a]";
  ffmpeg([...inputs, ...wavs.flatMap((w) => ["-i", w]),
    "-filter_complex", `${video};${audio}`, "-map", "[v]", "-map", "[a]", "-t", "21",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out]);
  return out;
}

async function submit(page, file, caption, posted) {
  await page.goto("/");
  await page.setInputFiles('input[type="file"]', file);
  await page.getByPlaceholder(/Paste the full caption/).fill(caption);
  if (posted) await page.locator('input[type="date"]').fill(posted);
  await page.getByRole("button", { name: /Run audit/ }).click();
  await expect(page.getByText("Audit result")).toBeVisible();
  await page.waitForTimeout(500);
}

test("the report's screenshots", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Choose an image or video" })).toBeVisible();
  await page.screenshot({ path: join(OUT, "screen01_upload_form.png") });

  await submit(page, IMAGE, IMAGE_CAPTION, POSTED);
  await page.screenshot({ path: join(OUT, "screen02_image_result.png"), fullPage: true });
  const finding = page.locator("article", { hasText: "Flag raised" }).first();
  await finding.locator("summary").click();
  // The opened finding just below the bar that keeps "Review another post" in view.
  await finding.evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 150));
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, "screen03_image_finding_opened.png") });

  await submit(page, makeVideo(), VIDEO_CAPTION, POSTED);
  const player = page.locator("video[controls]");
  await expect(player).toBeVisible();
  // Fully loaded, so the player shows its first frame and no loading spinner.
  await expect.poll(() => player.evaluate((v) => v.readyState)).toBe(4);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, "screen04_video_result.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.screenshot({ path: join(OUT, "screen05_phone_upload_form.png") });
});
