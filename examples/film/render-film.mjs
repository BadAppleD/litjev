// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Minimal Labs
// Adapted from jevlike examples/film/render-film.mjs @ 94f5fd1.
// Modified: explicit replay path, bounded duration, async image decode, safe process cleanup.
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {chromium} from 'playwright';

const [input, output, duration] = process.argv.slice(2);
if (!input || !output) throw Error('usage: node render-film.mjs REPLAY.html OUTPUT.mp4 [SECONDS]');
if (duration !== undefined && !(Number(duration) > 0 && Number.isFinite(Number(duration)))) throw Error('Invalid duration');
const browser = await chromium.launch({headless: true});
let ffmpeg;
try {
  const page = await browser.newPage({viewport: {width: 1920, height: 1080}, deviceScaleFactor: 1});
  await page.goto(pathToFileURL(path.resolve(input)).href + '?capture');
  await page.waitForFunction(() => Number.isFinite(window.filmDuration), {timeout: 10000});
  const seconds = Math.min(Number(duration || 60), await page.evaluate(() => window.filmDuration));
  // Refuse to overwrite an existing output; callers choose a fresh path.
  ffmpeg = spawn('ffmpeg', ['-n', '-loglevel', 'error', '-f', 'image2pipe', '-vcodec', 'png', '-r', '24', '-i', 'pipe:0', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path.resolve(output)], {stdio: ['pipe', 'inherit', 'inherit']});
  let failure;
  ffmpeg.on('error', error => {failure = error});
  ffmpeg.stdin.on('error', error => {failure = error});
  const done = new Promise(resolve => {ffmpeg.on('close', code => resolve(code)); ffmpeg.on('error', () => resolve(-1))});
  const total = Math.ceil(24 * seconds);
  for (let frame = 0; frame < total; frame++) {
    if (failure || ffmpeg.exitCode !== null) throw failure || Error(`ffmpeg exited ${ffmpeg.exitCode}`);
    await page.evaluate(t => window.setFrame(t), frame / 24);
    const png = await page.screenshot();
    if (!ffmpeg.stdin.write(png)) await once(ffmpeg.stdin, 'drain');
  }
  ffmpeg.stdin.end();
  const code = await done;
  if (code !== 0) throw failure || Error(`ffmpeg exited ${code}`);
  console.log(`wrote ${output} (${seconds.toFixed(2)} s of recorded wall time; no soundtrack)`);
} finally {
  if (ffmpeg && ffmpeg.exitCode === null) ffmpeg.kill();
  await browser.close();
}
