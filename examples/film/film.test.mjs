import assert from 'node:assert/strict';
import {test} from 'node:test';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {chromium} from 'playwright';

test('trace replay renders actual values and never executes action labels', async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(path.resolve('../../src/litjev/static/film.html')).href);
    const png = (await page.screenshot()).toString('base64');
    const trace = {
      schema_version: 'litjev.trace.v1', environment: 'test',
      actions: ['<img src=x onerror="window.injected=1">', 'right'],
      decisions: [0, 1].map(i => ({
        episode: 1, step: i, action_index: i, action: i ? 'right' : 'left',
        frame: `data:image/png;base64,${png}`, probabilities: [.25, .75],
        logits: [0, 1], provenance: {module: 'lm_head'}, timestamp: i + 1,
        latency_ms: 10, running_reward: 0, forward_calls: 2, output_tokens: 0,
        calibration_fitted: false, truncated: i === 1, terminated: false,
      })),
    };
    await page.locator('#file').setInputFiles({name:'trace.json', mimeType:'application/json', buffer:Buffer.from(JSON.stringify(trace))});
    await page.waitForFunction(() => window.filmDuration === 2);
    await page.evaluate(() => window.setFrame(1.5));
    assert.equal(await page.locator('#action').textContent(), 'right');
    assert.match(await page.locator('#status').textContent(), /2 forwards.*0 generated tokens.*truncated/);
    assert.equal(await page.locator('.prob img').count(), 0);
    assert.equal(await page.evaluate(() => window.injected), undefined);
    assert.equal(await page.locator('progress').nth(1).getAttribute('value'), '0.75');
    assert.equal(await page.evaluate(() => document.querySelector('#frame').naturalWidth > 0), true);
  } finally {
    await browser.close();
  }
});
