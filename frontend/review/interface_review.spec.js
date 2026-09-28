// Automated measures and the walkthrough pack for the interface review. Nothing here judges the
// interface: it records what the screens show and measure, for the reviewer. Run with
// `npm run review:interface` (ROUND=A or B). Needs datasets A and E on this machine and writes
// to results/, which is not in the repository.
//
//   results/interface_review_<ROUND>.json      the automated measures
//   results/interface_review/round_<ROUND>/    T1-T5 at 1366 x 768 (one numbered screenshot per
//                                              step), captions.csv, index.html, and an empty
//                                              round_<ROUND>.csv for the reviewer to fill in
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ROUND = process.env.ROUND ?? "A";
const ROOT = resolve("..");
const OUT_JSON = join(ROOT, "results", `interface_review_${ROUND}.json`);
const PACK = join(ROOT, "results", "interface_review", `round_${ROUND}`);
const WORK = join(PACK, "_materials");
const PYTHON = join(ROOT, "backend", process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python");
const SIZES = { desktop: { width: 1366, height: 768 }, phone: { width: 390, height: 844 } };

// The posts the tasks use: a dated NASA photo that is in the image history index, a narrated clip
// with one line that describes a different scene, and a screenshot of a text post.
const PHOTO = join(ROOT, "data", "A_originals", "A01_KSC-2013-2815.jpg");
const PHOTO_CAPTION = "Storm clouds gather over the space centre this afternoon.";
const PHOTO_DATE = "2024-06-01";
const CLIP = join(ROOT, "data", "E_videos", "clips", "E09.mp4");
const CLIP_CAPTION = "A hurricane seen from the space station";
const TEXT_POST_CAPTION = "Read this before it disappears";

// Keystroke-Level Model operators in seconds (Card, Moran and Newell, 1980). K is an average
// non-secretarial typist; a click is B B (press, release). R, the system's response, is measured
// on each run. The model has no operator for scrolling, so that goes under controls_visible.
const KLM = { K: 0.28, P: 1.1, B: 0.1, H: 0.4, M: 1.35 };
const KLM_NOTES = {
  K: "keystroke (average non-secretarial typist)",
  P: "point with the mouse",
  B: "mouse button press or release (a click is B B)",
  H: "move a hand between mouse and keyboard",
  M: "mental preparation",
  R: "system response, measured",
};

const VERDICT_WORDS = ["safe", "unsafe", "fake", "true", "false", "verified", "unverified", "real", "genuine",
  "authentic", "trustworthy", "untrustworthy", "hoax", "debunked", "misinformation", "disinformation",
  "correct", "incorrect", "accurate", "inaccurate", "legit", "lie", "lies", "trust", "reliable",
  "unreliable", "credible", "proven", "confirmed"];
const VERDICT = new RegExp(`\\b(${VERDICT_WORDS.join("|")})\\b`, "gi");

const results = {
  round: ROUND,
  generated: new Date().toISOString(),
  method: "docs/INTERFACE_REVIEW.md section 5; nothing in this file is a judgement of the interface",
  viewports: SIZES,
  accessibility: {},
  click_targets: {},
  text_size: {},
  steps_per_task: {},
  predicted_time: { operators_s: KLM, operators: KLM_NOTES, tasks: {} },
  controls_visible: { desktop: {}, phone: {} },
  no_verdict_wording: { words: VERDICT_WORDS, ui_hits: [], source_hits: [], backend: null },
  reading_level: null,
};
const packSteps = [];
const screenShots = [];

function ffmpeg(args) {
  execFileSync("ffmpeg", ["-v", "error", "-y", ...args]);
}

function materials() {
  mkdirSync(WORK, { recursive: true });
  const font = process.platform === "win32" ? ":fontfile='C\\:/Windows/Fonts/arialbd.ttf'" : "";
  // Words cover more than 40% of this image (45% measured), so the picture check calls it
  // mostly text.
  const lines = ["BREAKING NEWS TODAY", "THE BRIDGE IS CLOSED", "SHARE THIS WITH ALL", "YOUR FAMILY NOW",
    "BEFORE THEY DELETE", "THIS POST FOREVER"];
  const text = lines.map((l, i) =>
    `drawtext=text='${l}'${font}:fontsize=70:fontcolor=black:x=18:y=${16 + i * 88}`).join(",");
  const textPost = join(WORK, "text_post.png");
  ffmpeg(["-f", "lavfi", "-i", "color=c=white:s=900x540", "-vf", text, "-frames:v", "1", textPost]);
  const longClip = join(WORK, "long_clip.mp4");
  ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x240:r=5:d=65", "-c:v", "libx264", "-pix_fmt", "yuv420p", longClip]);
  const note = join(WORK, "notes.txt");
  writeFileSync(note, "not a picture");
  return { textPost, longClip, note };
}

// --- measures taken on a screen ------------------------------------------------------------------

async function measureScreen(page, name, caption) {
  // Full-page screenshot for the heuristic review. Taken first so a short-lived state such as
  // loading is still on screen.
  const file = `S${String(screenShots.length + 1).padStart(2, "0")}.png`;
  await page.screenshot({ path: join(PACK, file), fullPage: true });
  screenShots.push({ file, name, caption });
  const axe = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze();
  const counts = { critical: 0, serious: 0, moderate: 0, minor: 0 };
  for (const v of axe.violations) counts[v.impact] = (counts[v.impact] ?? 0) + v.nodes.length;
  results.accessibility[name] = {
    counts_by_impact: counts,
    violations: axe.violations.map((v) => ({
      id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.length,
      targets: v.nodes.slice(0, 8).map((n) => n.target.join(" ")),
    })),
  };

  const targets = await page.evaluate(() => {
    const out = [];
    const sel = "button, a[href], input:not([type=hidden]), textarea, select, summary, [role=button]";
    for (const el of document.querySelectorAll(sel)) {
      const r = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden" || (r.width === 0 && r.height === 0)) continue;
      const label = (el.getAttribute("aria-label") || el.innerText || el.value || el.getAttribute("title")
        || el.getAttribute("type") || el.tagName).trim().replace(/\s+/g, " ").slice(0, 60);
      out.push({ element: el.tagName.toLowerCase(), label, width: Math.round(r.width), height: Math.round(r.height) });
    }
    return out;
  });
  results.click_targets[name] = {
    controls: targets.length,
    under_24x24: targets.filter((t) => t.width < 24 || t.height < 24),
    under_44x44: targets.filter((t) => t.width < 44 || t.height < 44).length,
    all: targets,
  };

  const text = await page.evaluate(() => {
    const sizes = {};
    let chars = 0;
    let small = 0;
    const smallest = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const value = node.textContent.trim();
      const el = node.parentElement;
      if (!value || !el || ["SCRIPT", "STYLE", "OPTION"].includes(el.tagName)) continue;
      const r = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden" || r.width === 0) continue;
      if (el.closest("details:not([open])") && !el.closest("summary")) continue;
      const px = parseFloat(style.fontSize);
      sizes[px] = (sizes[px] ?? 0) + value.length;
      chars += value.length;
      if (px < 16) {
        small += value.length;
        if (smallest.length < 40) smallest.push({ px, text: value.slice(0, 50) });
      }
    }
    return { chars, chars_under_16px: small, share_under_16px: chars ? +(small / chars).toFixed(3) : 0,
      chars_by_px: sizes, examples_under_16px: smallest };
  });
  results.text_size[name] = text;

  const ui = await page.evaluate(() => document.body.innerText);
  const userContent = [PHOTO_CAPTION, CLIP_CAPTION, TEXT_POST_CAPTION];
  let scanned = ui;
  for (const c of userContent) scanned = scanned.split(c).join(" ");
  for (const m of scanned.matchAll(VERDICT)) {
    results.no_verdict_wording.ui_hits.push({
      screen: name, word: m[0], context: scanned.slice(Math.max(0, m.index - 60), m.index + 60).replace(/\s+/g, " "),
    });
  }
}

// --- the scripted tasks ----------------------------------------------------------------------------

// How far a control is from the visible screen: 0 = visible now, n = n screens down, -n = n up.
async function scrollsAway(locator) {
  return locator.evaluate((el) => {
    const r = el.getBoundingClientRect();
    const h = window.innerHeight;
    if (r.top >= 0 && r.bottom <= h) return 0;
    if (r.top < 0) return -Math.ceil(-r.top / h);
    return Math.max(1, Math.ceil((r.bottom - h) / h));
  });
}

async function outline(locator, on) {
  await locator.evaluate((el, show) => {
    el.style.outline = show ? "4px solid #d6008f" : "";
    el.style.outlineOffset = show ? "3px" : "";
  }, on);
}

// Runs one task's steps in order, measuring each step's control before acting on it.
async function runTask(page, size, task) {
  const record = [];
  let shot = 0;
  for (const step of task.steps) {
    const target = step.target ? step.target(page) : null;
    let away = null;
    if (target) {
      await expect(target).toBeVisible({ timeout: 180_000 });
      away = await scrollsAway(target);
      await target.scrollIntoViewIfNeeded();
    }
    if (size === "desktop") {
      shot += 1;
      const file = `${task.id}-${String(shot).padStart(2, "0")}.png`;
      if (target) await outline(target, true);
      await page.screenshot({ path: join(PACK, file) });
      if (target) await outline(target, false);
      const caption = (away > 0 ? "Scroll down, then: " : away < 0 ? "Scroll up, then: " : "") + step.say;
      packSteps.push({ task: task.id, step: shot, file, caption });
    }
    let response = null;
    if (step.act) {
      const started = Date.now();
      await step.act(page, target);
      if (step.waitFor) {
        await expect(step.waitFor(page)).toBeVisible({ timeout: 180_000 });
        response = (Date.now() - started) / 1000;
      }
      await page.waitForTimeout(800); // let any scrolling the page does itself finish
    }
    record.push({ step: step.say, klm: step.klm, scrolls_needed: away === null ? null : Math.abs(away),
      direction: away > 0 ? "down" : away < 0 ? "up" : null, response_s: response });
  }
  if (size === "desktop" && task.result) {
    shot += 1;
    const file = `${task.id}-${String(shot).padStart(2, "0")}.png`;
    const target = task.result.target(page);
    await target.scrollIntoViewIfNeeded();
    await outline(target, true);
    await page.screenshot({ path: join(PACK, file) });
    await outline(target, false);
    packSteps.push({ task: task.id, step: shot, file, caption: task.result.say });
  }
  results.controls_visible[size][task.id] = record.map((r) => ({ step: r.step, scrolls_needed: r.scrolls_needed,
    direction: r.direction }));
  if (size === "desktop") {
    const ops = record.flatMap((r) => r.klm.split(/\s+/).filter(Boolean));
    const responses = record.filter((r) => r.response_s != null).map((r) => r.response_s);
    const seconds = ops.reduce((s, op) => s + KLM[op], 0) + responses.reduce((s, r) => s + r, 0);
    results.steps_per_task[task.id] = {
      steps: task.steps.length,
      actions: task.steps.filter((s) => s.act).length,
      description: task.title,
    };
    results.predicted_time.tasks[task.id] = {
      operators: ops.join(" ") + (responses.length ? ` + R(${responses.map((r) => r.toFixed(1)).join(", ")})` : ""),
      operator_counts: ops.reduce((c, op) => ({ ...c, [op]: (c[op] ?? 0) + 1 }), {}),
      response_s: responses,
      seconds: +seconds.toFixed(2),
    };
  }
}

// The grey box: the form's first plain button (its name changes once a file is chosen).
const dropzone = (p) => p.locator("form button[type=button]").first();
const captionBox = (p) => p.getByPlaceholder(/Paste the full caption/);
const runButton = (p) => p.getByRole("button", { name: /Run audit/ });
const card = (p, label) => p.locator("article", { hasText: label });

function chooseSteps(file, kind) {
  return [
    { say: `Click the grey box to choose the ${kind}.`, klm: "M P B B", target: dropzone,
      act: async (p, t) => {
        const chooser = p.waitForEvent("filechooser");
        await t.click();
        await (await chooser).setFiles(file);
      } },
    { say: `In the file window that opens, double-click the ${kind} (the window is not shown here).`,
      klm: "M P B B B B", target: dropzone },
  ];
}

function tasks(files) {
  const paste = (text) => ({
    say: "Click the caption box and paste the post's caption.", klm: "M P B B H K K", target: captionBox,
    act: async (p, t) => t.fill(text),
  });
  const run = (label) => ({
    say: `Click "Run audit"${label ? ` and wait (${label})` : ""}.`, klm: "H M P B B", target: runButton,
    act: async (p, t) => t.click(), waitFor: (p) => p.getByText("Audit result"),
  });
  return {
    T1: {
      id: "T1", title: "Check an image post with Aegis", start: "/",
      steps: [...chooseSteps(PHOTO, "photo"), paste(PHOTO_CAPTION), run(),
        { say: "Read the summary at the top of the result.", klm: "M",
          target: (p) => p.getByText(/checks run\./) }],
    },
    T2: {
      id: "T2", title: "Find what Aegis suggests checking before sharing", start: "after T1",
      steps: [
        { say: 'Find the first finding marked "Flag raised".', klm: "M",
          target: (p) => p.locator("article", { hasText: "Flag raised" }).first() },
        { say: 'Read "What to check" in that finding.', klm: "M",
          target: (p) => p.locator("article", { hasText: "Flag raised" }).first().getByText("What to check", { exact: true }) },
      ],
      result: { say: "What Aegis suggests checking, in the finding that raised a flag.",
        target: (p) => p.locator("article", { hasText: "Flag raised" }).first() },
    },
    T3: {
      id: "T3", title: "Add the date the post claims and check again, then see what changed", start: "after T1",
      steps: [
        { say: 'Click "Review another post".', klm: "M P B B",
          target: (p) => p.getByRole("button", { name: /Review another post/ }), act: async (p, t) => t.click() },
        ...chooseSteps(PHOTO, "photo again"),
        paste(PHOTO_CAPTION),
        { say: 'Click the "Date posted" box and type the date the post claims.', klm: "H M P B B H K K K K K K K K",
          target: (p) => p.locator('input[type="date"]'), act: async (p, t) => t.fill(PHOTO_DATE) },
        run(),
        { say: 'Find the "Recycled context" finding and read what it now says about the date.', klm: "M",
          target: (p) => card(p, "Recycled context") },
      ],
    },
    T4: {
      id: "T4", title: "In a video post, find the moment where what's said doesn't match what's shown", start: "/",
      steps: [...chooseSteps(CLIP, "video"), paste(CLIP_CAPTION), run("a video takes up to a minute"),
        { say: 'Find the "Speech ↔ picture match" finding marked "Flag raised".', klm: "M",
          target: (p) => card(p, "Speech ↔ picture match") },
        { say: "Click the moment button (▶ and a time) in that finding.", klm: "M P B B",
          target: (p) => card(p, "Speech ↔ picture match").locator('button[title^="Play from"]').first(),
          act: async (p, t) => t.click() },
        { say: "The video player has moved to that moment: watch and listen there.", klm: "M",
          target: (p) => p.locator("video[controls]") },
      ],
    },
    T5: {
      id: "T5", title: 'Work out what a "Couldn\'t check" result means', start: "a text screenshot checked",
      setup: { file: files.textPost, caption: TEXT_POST_CAPTION },
      steps: [
        { say: 'Find the finding marked "Couldn\'t check".', klm: "M",
          target: (p) => p.locator("article", { hasText: "Couldn’t check" }).first() },
        { say: 'Read its explanation and the line starting "Why:".', klm: "M",
          target: (p) => p.locator("article", { hasText: "Couldn’t check" }).first().getByText("Why:") },
        { say: 'Read "What to check yourself" below it.', klm: "M",
          target: (p) => p.locator("article", { hasText: "Couldn’t check" }).first().getByText("What to check yourself") },
      ],
    },
  };
}

async function freshStart(page) {
  await page.goto("/");
  await expect(page.getByText("Submit a post for review")).toBeVisible();
}

async function submit(page, file, caption) {
  await freshStart(page);
  await page.setInputFiles('input[type="file"]', file);
  await captionBox(page).fill(caption);
  await runButton(page).click();
  await expect(page.getByText("Audit result")).toBeVisible({ timeout: 180_000 });
}

async function runAllTasks(page, size, files) {
  await page.setViewportSize(SIZES[size]);
  const all = tasks(files);
  await freshStart(page);
  await runTask(page, size, all.T1);
  await runTask(page, size, all.T2);
  await runTask(page, size, all.T3);
  await freshStart(page);
  await runTask(page, size, all.T4);
  await submit(page, all.T5.setup.file, all.T5.setup.caption);
  await page.evaluate(() => window.scrollTo(0, 0));
  await runTask(page, size, all.T5);
}

// --- the run -----------------------------------------------------------------------------------------

test.describe.configure({ mode: "serial" });
test.skip(!existsSync(PHOTO) || !existsSync(CLIP), "datasets A and E are needed on this machine");

let files;
test.beforeAll(() => {
  mkdirSync(PACK, { recursive: true });
  files = materials();
});

test("walkthrough of T1 to T5 at 1366 x 768: the pack, steps and predicted times", async ({ page }) => {
  test.setTimeout(1_800_000);
  await runAllTasks(page, "desktop", files);
});

test("T1 to T5 at 390 x 844: controls visible without scrolling", async ({ page }) => {
  test.setTimeout(1_800_000);
  await runAllTasks(page, "phone", files);
});

test("every screen in section 4: accessibility, targets, text size and wording", async ({ page }) => {
  test.setTimeout(1_800_000);
  await page.setViewportSize(SIZES.desktop);
  await freshStart(page);
  await measureScreen(page, "upload form", "The upload form, as it first appears.");
  await page.setInputFiles('input[type="file"]', PHOTO);
  await captionBox(page).fill(PHOTO_CAPTION);
  await expect(page.getByText(/The image will be sent to Google/)).toBeVisible();
  await measureScreen(page, "upload form, photo chosen (with the live-lookup notice)",
    "The form with a photo chosen and a caption pasted: the web-search option and its notice appear.");

  await page.route("**/analyze", async (route) => {
    await new Promise((r) => setTimeout(r, 4000));
    await route.continue();
  });
  await runButton(page).click();
  await expect(page.getByRole("button", { name: /Running audit/ })).toBeVisible();
  await measureScreen(page, "loading", 'Loading: after "Run audit" is clicked, while the result is prepared.');
  await expect(page.getByText("Audit result")).toBeVisible({ timeout: 180_000 });
  await page.unroute("**/analyze");
  await measureScreen(page, "image scorecard", "The result for an image post.");
  await page.locator("article", { hasText: "Flag raised" }).first().locator("summary").click();
  await measureScreen(page, "an opened flag (supporting evidence shown)",
    'A finding opened: "Supporting evidence" shown under the flag that was raised.');

  await submit(page, files.textPost, TEXT_POST_CAPTION);
  await measureScreen(page, "a \"Couldn't check\" result",
    "A result with a \"Couldn't check\" finding (the post is a screenshot of a text post).");

  await submit(page, CLIP, CLIP_CAPTION);
  await measureScreen(page, "video scorecard", "The result for a video post.");

  await freshStart(page);
  await page.setInputFiles('input[type="file"]', files.note);
  await measureScreen(page, "error: wrong file type (a text file chosen)",
    "A text file chosen instead of a picture or video: what the page shows then.");

  await freshStart(page);
  await page.setInputFiles('input[type="file"]', files.longClip);
  await runButton(page).click();
  await expect(page.getByText(/Audit could not run/)).toBeVisible({ timeout: 60_000 });
  await measureScreen(page, "error: video too long", 'A 65-second video after "Run audit": the message shown.');

  await freshStart(page);
  await page.setInputFiles('input[type="file"]', PHOTO);
  await page.route("**/analyze", (route) => route.abort());
  await runButton(page).click();
  await expect(page.getByText(/Audit could not run/)).toBeVisible();
  await measureScreen(page, "error: backend down", 'The server cannot be reached, after "Run audit": the message shown.');
  await page.unroute("**/analyze");

  // Interface wording in the source, and the backend's own lines (reading level, verdict words).
  const src = join(resolve("."), "src");
  const sources = [join(src, "App.jsx"), join(src, "api.js"),
    ...readdirSync(join(src, "components")).map((f) => join(src, "components", f))];
  for (const file of sources) {
    readFileSync(file, "utf-8").split(/\r?\n/).forEach((line, i) => {
      if (/^\s*(\/\/|\/\*|\*)/.test(line)) return; // comments are not shown to anyone
      for (const m of line.matchAll(VERDICT)) {
        results.no_verdict_wording.source_hits.push({
          file: file.slice(ROOT.length + 1).replace(/\\/g, "/"), line: i + 1, word: m[0], context: line.trim().slice(0, 140),
        });
      }
    });
  }
  const audit = JSON.parse(execFileSync(PYTHON, [join(ROOT, "backend", "scripts", "interface_text_audit.py")],
    { cwd: ROOT, env: { ...process.env, HF_HUB_OFFLINE: "1" }, maxBuffer: 64 * 1024 * 1024 }).toString("utf-8"));
  results.reading_level = audit.reading_level;
  results.no_verdict_wording.backend = audit.backend_verdict_words;
});

test.afterAll(() => {
  writeFileSync(OUT_JSON, JSON.stringify(results, null, 1));

  // The walkthrough pack: captions, a page to view it, and the empty recording sheet.
  const csv = (v) => `"${String(v).replace(/"/g, '""')}"`;
  writeFileSync(join(PACK, "captions.csv"),
    ["task,step,file,caption", ...packSteps.map((s) => [s.task, s.step, s.file, csv(s.caption)].join(",")),
      ...screenShots.map((s, i) => ["screens", i + 1, s.file, csv(`${s.name}: ${s.caption}`)].join(","))].join("\n") + "\n");
  const titles = Object.fromEntries(Object.values(tasks(files)).map((t) => [t.id, t.title]));
  const html = ["<!doctype html><meta charset='utf-8'><title>Walkthrough, round " + ROUND + "</title>",
    "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 16px}",
    "img{width:100%;border:1px solid #ccc}figure{margin:0 0 36px}figcaption{font-weight:600;margin:6px 0}</style>",
    `<h1>Walkthrough pack, round ${ROUND}</h1><p>1366 x 768. The pink outline marks the control each step uses.</p>`];
  for (const id of ["T1", "T2", "T3", "T4", "T5"]) {
    html.push(`<h2>${id}. ${titles[id]}</h2>`);
    for (const s of packSteps.filter((p) => p.task === id)) {
      html.push(`<figure><figcaption>${id}-${s.step}. ${s.caption.replace(/</g, "&lt;")}</figcaption><img src="${s.file}" alt=""></figure>`);
    }
  }
  html.push("<h2>Screens for the heuristic review</h2><p>Every screen in section 4 of the review plan, "
    + "whole page at 1366 wide, as the automated measures saw it.</p>");
  screenShots.forEach((s, i) => {
    html.push(`<figure><figcaption>S${i + 1}. ${s.name}: ${s.caption.replace(/</g, "&lt;")}</figcaption>`
      + `<img src="${s.file}" alt=""></figure>`);
  });
  writeFileSync(join(PACK, "index.html"), html.join("\n"));
  const sheet = join(PACK, `round_${ROUND}.csv`);
  if (!existsSync(sheet)) writeFileSync(sheet, "id,method,where,check,problem,severity\n");
});
