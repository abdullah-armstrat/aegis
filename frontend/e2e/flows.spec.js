// The image and video flows end to end, in a real browser against the real backend.
// Media is made with ffmpeg in a temporary folder, so no third-party files are needed; the last
// test uses a narrated dataset E clip when it is present on this machine and is skipped otherwise.
import { test, expect } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const work = mkdtempSync(join(tmpdir(), "aegis-e2e-"));
const FONT =
  process.platform === "win32" ? ":fontfile='C\\:/Windows/Fonts/arialbd.ttf'" : "";
const NARRATED_CLIP = resolve("..", "data", "E_videos", "clips", "E09.mp4");

function ffmpeg(args) {
  execFileSync("ffmpeg", ["-v", "error", "-y", ...args]);
}

// Three 3-second scenes: red, blue with pressure wording burned in, green. No audio track.
function makeVideo(path) {
  const text =
    `drawtext=text='SHARE NOW BEFORE THEY DELETE IT!!'${FONT}:fontsize=26:fontcolor=white:` +
    "x=(w-text_w)/2:y=(h-text_h)/2";
  ffmpeg([
    "-f", "lavfi", "-i", "color=c=red:s=640x360:r=10:d=3",
    "-f", "lavfi", "-i", "color=c=blue:s=640x360:r=10:d=3",
    "-f", "lavfi", "-i", "color=c=green:s=640x360:r=10:d=3",
    "-filter_complex", `[1:v]${text}[b];[0:v][b][2:v]concat=n=3:v=1:a=0[v]`,
    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
  ]);
  return path;
}

function parseClock(label) {
  const m = label.match(/(\d+):(\d\d)/);
  return Number(m[1]) * 60 + Number(m[2]);
}

async function upload(page, file, caption) {
  await page.goto("/");
  await page.setInputFiles('input[type="file"]', file);
  if (caption) await page.getByPlaceholder(/Paste the full caption/).fill(caption);
  const preview = page.locator("form video");
  if (await preview.count()) {
    // The chosen video is previewed as a frame, not an empty box.
    await expect.poll(() => preview.evaluate((v) => v.videoWidth)).toBeGreaterThan(0);
  }
  await page.getByRole("button", { name: /Run audit/ }).click();
}

// Every time button (keyframe strip and findings) must move the player to its moment.
async function expectEveryTimeButtonSeeks(page) {
  const player = page.locator("video[controls]");
  await expect(player).toBeVisible();
  const buttons = page.locator('button[title^="Go to"], button[title^="Play from"]');
  const count = await buttons.count();
  expect(count).toBeGreaterThan(0);
  for (let i = 0; i < count; i++) {
    const button = buttons.nth(i);
    const target = parseClock(await button.innerText());
    await button.click();
    await expect
      .poll(async () => player.evaluate((v) => v.currentTime), { timeout: 10_000 })
      .toBeGreaterThanOrEqual(target - 0.6);
    const now = await player.evaluate((v) => v.currentTime);
    expect(now).toBeLessThanOrEqual(target + 0.6);
  }
  return count;
}

test("image flow: upload, results, and the reason a check could not run", async ({ page }) => {
  const image = join(work, "harbour.png");
  ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=640x480", "-frames:v", "1", image]);
  await upload(page, image, "");

  await expect(page.getByText("Audit result")).toBeVisible();
  await expect(page.getByText(/Findings \(3\)/)).toBeVisible();
  // No caption: the wording check could not run, and the card says why.
  const framing = page.locator("article", { hasText: "Shouting style" });
  await expect(framing.getByText("Why:")).toBeVisible();
  await expect(framing).toContainText("There is no caption text to check");
  // An image has no player and no time buttons.
  await expect(page.locator("video")).toHaveCount(0);
  await expect(page.locator('button[title^="Play from"]')).toHaveCount(0);
});

test("video flow: player, keyframe strip, reasons, and every time button seeks", async ({ page }) => {
  const video = makeVideo(join(work, "three-scenes.mp4"));
  await upload(page, video, "A quiet harbour at dawn");

  await expect(page.getByText("Audit result")).toBeVisible();
  await expect(page.getByText(/Findings \(4\)/)).toBeVisible();
  await expect(page.getByText(/Keyframes checked \(3\)/)).toBeVisible();
  // The count sentence starts each sentence with a capital.
  await expect(page.getByText(/^Four checks run\. [A-Z]/)).toBeVisible();
  // No audio track: the speech check could not run, and the card says why.
  const speech = page.locator("article", { hasText: "Speech ↔ picture match" });
  await expect(speech.getByText("Why:")).toBeVisible();
  await expect(speech).toContainText("the video has no audio track");
  // The burned-in pressure wording is found in the frame that shows it.
  const framing = page.locator("article", { hasText: "Shouting style" });
  await expect(framing).toContainText("Some of the words in this video");
  await expect(framing.getByRole("button", { name: /0:05/ })).toBeVisible();

  const seeks = await expectEveryTimeButtonSeeks(page);
  expect(seeks).toBeGreaterThanOrEqual(5); // 3 keyframes + caption moment + framing moment
});

test("video over 60 seconds is refused with the limit named", async ({ page }) => {
  const long = join(work, "long.mp4");
  ffmpeg(["-f", "lavfi", "-i", "color=c=gray:s=160x120:r=5:d=61", "-c:v", "libx264", "-pix_fmt", "yuv420p", long]);
  await upload(page, long, "");
  await expect(page.getByText(/Audit could not run: The video is 61\.0 s long; the limit is 60 s\./)).toBeVisible();
});

test("narrated clip: the speech finding's moments move the player", async ({ page }) => {
  test.skip(!existsSync(NARRATED_CLIP), "dataset E is not on this machine");
  await upload(page, NARRATED_CLIP, "A hurricane seen from the space station");

  const speech = page.locator("article", { hasText: "Speech ↔ picture match" });
  await expect(speech).toBeVisible();
  await expect(speech.locator('button[title^="Play from"]').first()).toBeVisible();
  // All eight keyframes fit inside the strip, none cut off at the edge.
  const strip = page.getByText(/Keyframes checked \(8\)/).locator("xpath=..");
  const box = await strip.boundingBox();
  const frames = strip.locator('button[title^="Go to"]');
  await expect(frames).toHaveCount(8);
  for (let i = 0; i < 8; i++) {
    const b = await frames.nth(i).boundingBox();
    expect(b.x + b.width).toBeLessThanOrEqual(box.x + box.width + 1);
  }
  await expectEveryTimeButtonSeeks(page);
});

test("the web-search option is offered unticked for images only", async ({ page }) => {
  const image = join(work, "option.png");
  ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x240", "-frames:v", "1", image]);
  await page.goto("/");
  const option = page.getByRole("checkbox", { name: /Also search the web/ });
  await expect(option).toHaveCount(0); // nothing chosen yet
  await page.setInputFiles('input[type="file"]', image);
  await expect(option).toBeVisible();
  await expect(option).not.toBeChecked();
  await expect(page.getByText(/The image will be sent to Google/)).toBeVisible();
  await page.setInputFiles('input[type="file"]', makeVideo(join(work, "option.mp4")));
  await expect(option).toHaveCount(0); // videos use the local index only
});
