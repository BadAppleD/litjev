import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';

test('playground sends canonical questions and renders choice, score, noul and timing', async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let sent;
    await page.route('http://litjev.test/**', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/health') return route.fulfill({json: {model_loaded: true}});
      if (url.pathname === '/v1/systemone/debug') {
        sent = route.request().postDataJSON();
        return route.fulfill({json: {
          result: {model: 'fixture', usage: {input_tokens: 42, output_tokens: 0}, answers: {
            q1: {type: 'choice', choice: 'B', probabilities: {A: .1, B: .9}, confidence: .64},
            q2: {type: 'score', score: .7, legend: {'0': 'low', '1': 'high'}, probabilities: {'0': .3, '1': .7}, confidence: .16},
            q3: {type: 'noul', noul: .8},
          }},
          diagnostics: {forward_calls: 2, calibration_fitted: false,
            timing: {decision_seconds: .123, model_setup_seconds: .001},
            fields: Object.fromEntries(['q1', 'q2', 'q3'].map(k => [k, {provenance: {}, probabilities: {false: .2, true: .8}}]))},
        }});
      }
      const name = url.pathname === '/' ? 'index.html' : url.pathname === '/example' ? 'example.json' : path.basename(url.pathname);
      const contentType = name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : name.endsWith('.json') ? 'application/json' : 'text/html';
      return route.fulfill({body: await readFile(path.resolve('../../src/litjev/static', name)), contentType});
    });
    await page.goto('http://litjev.test/');
    await page.waitForFunction(() => document.querySelector('#schema-input').value.includes('q10'));
    await page.locator('#submit').click();
    await page.waitForFunction(() => document.querySelector('#result-status').textContent.includes('3 个问题'));
    assert.deepEqual(Object.keys(sent).sort(), ['model', 'questions', 'state']);
    assert.equal(Object.keys(sent.questions).length, 10);
    assert.equal(sent.questions.q1.type, 'choice');
    assert.equal(await page.locator('#decision-time').textContent(), '0.123 s');
    assert.match(await page.locator('#answers').textContent(), /B.*0.7000.*P\(yes\) 0.8000/s);
    assert.match(await page.locator('#raw').textContent(), /"diagnostics"/);
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
});
