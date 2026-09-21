// SPDX-License-Identifier: Apache-2.0
// Headless check of the bundled viewer: no page errors, and the numbers the
// plan's acceptance criteria name (VIEWER_PLAN.md). Run through test/run.sh.
//
//   NODE_PATH=$(npm root -g) node check.js <bundled.html> <fixtures dir>
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('playwright');

const [, , page_path, fixtures] = process.argv;
const EXE = process.env.CHROMIUM || '/opt/pw-browsers/chromium';
let failures = 0;
const ok = (cond, msg) => { console.log((cond ? 'ok   ' : 'FAIL ') + msg); if (!cond) failures++; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;

(async () => {
  const browser = await chromium.launch({ executablePath: EXE, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('file://' + path.resolve(page_path));
  await page.waitForFunction(() => window.__nntr && window.__nntr.model());
  const names = await page.evaluate(() => window.__nntr.embedded());
  ok(names.length >= 2, 'embedded traces: ' + names.join(' | '));

  const use = async name => { await page.selectOption('#trace-select', name); await page.waitForTimeout(50); };
  const asBuilt = names.find(n => /as built/.test(n)), pipelined = names.find(n => /pipelined/.test(n));

  // W0: metrics on both samples
  await use(asBuilt);
  let m = await page.evaluate(() => window.__nntr.metrics());
  ok(near(m.compression, 1.17, 0.03), `as-built compression ${m.compression.toFixed(3)} ≈ 1.17`);
  ok(m.overlap === 0, 'as-built has no HMX∥HVX overlap');
  await use(pipelined);
  m = await page.evaluate(() => window.__nntr.metrics());
  ok(near(m.compression, 1.45, 0.05), `pipelined compression ${m.compression.toFixed(3)} ≈ 1.45`);
  ok(m.overlap > 0, 'pipelined has HMX∥HVX overlap');

  // W1: units-busy strip. While an FC HMX chunk runs, as-built has at most HMX+DMA
  // busy (2); the pipelined projection has HVX dequanting alongside (>= 3).
  const busyDuringHmx = () => page.evaluate(() => {
    const T = window.__nntr.model(), c = T.models[0].counters.find(c => c.kind === 'heat' && c.name === 'units busy');
    let mx = 0;
    for (const e of T.all) if (e.cat === 'dsp.hmx' && e.args && e.args.op === 'qkv_proj' && e.args.chunk > 0) {
      for (let k = 0; k < 8; k++) mx = Math.max(mx, window.__nntr.counterAt(c, e.ts + e.dur * (k + .5) / 8));
    }
    return { mx, max: c.max };
  });
  await use(asBuilt); let b = await busyDuringHmx();
  ok(b.mx <= 2, `as-built units busy during qkv HMX chunks ${b.mx} <= 2 (strip max ${b.max})`);
  await use(pipelined); b = await busyDuringHmx();
  ok(b.mx >= 3, `pipelined units busy during qkv HMX chunks ${b.mx} >= 3`);

  // W13: VTCM counter carries a budget and its hi-water stays under it
  await use(asBuilt);
  const vt = await page.evaluate(() => { const T = window.__nntr.model(), c = T.models[0].counters.find(c => c.name === 'VTCM (KB)'); return { budget: (T.md.budgets || {})[c.name], peak: Math.max(...c.pts.map(p => p[1])) }; });
  ok(vt.budget === 8192 && vt.peak < vt.budget, `VTCM hi-water ${vt.peak} KB under budget ${vt.budget} KB`);

  // W0: every tab renders, range select works
  await use(asBuilt);
  for (const id of await page.$$eval('#tabs button', bs => bs.map(b => b.dataset.tab))) {
    await page.evaluate(id => window.__nntr.tab(id), id);
    const txt = await page.$eval('#pane', e => e.innerText.trim());
    ok(txt.length > 0, `tab ${id} renders`);
  }
  await page.selectOption('#range-select', { index: 1 });
  const label = await page.$eval('#range-label', e => e.textContent);
  ok(/prefill/.test(label), 'range select -> prefill: ' + label);

  // dump metrics JSON for the python cross-check (W12)
  const mj = await page.evaluate(() => window.__nntr.metricsJSON());
  fs.mkdirSync(fixtures, { recursive: true });
  fs.writeFileSync(path.join(fixtures, 'viewer_metrics.json'), JSON.stringify(mj, null, 1));

  ok(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' / ') : ''));
  await page.screenshot({ path: path.join(fixtures, 'viewer.png') });
  await browser.close();
  console.log(failures ? `${failures} check(s) FAILED` : 'all checks passed');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
