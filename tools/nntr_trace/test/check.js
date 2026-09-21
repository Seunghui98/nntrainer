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

  // W2: `m` on a selected layer slice makes it the range; a ruler drag selects a range
  await page.evaluate(() => { const T = window.__nntr.model(); window.__nntr.select(T.all.find(e => e.cat === 'host.layer' && e.name === 'layer0')); });
  await page.keyboard.press('m');
  let w2 = await page.evaluate(() => {
    const T = window.__nntr.model(), e = T.all.find(x => x.cat === 'host.layer' && x.name === 'layer0');
    const a = window.__nntr.metrics(), b = window.__nntr.metrics({ t0: e.ts, t1: e.ts + e.dur });
    return { label: document.querySelector('#range-label').textContent, same: Math.abs(a.cpu - b.cpu) < 1e-6 && Math.abs(a.hmxOnly - b.hmxOnly) < 1e-6, sel: document.querySelector('#range-select').value };
  });
  ok(/^layer0/.test(w2.label) && w2.same && w2.sel === 'sel', `m -> range = layer0 (${w2.label})`);
  await page.keyboard.press('Escape');
  const cvb = await (await page.$('#cv')).boundingBox();
  await page.mouse.move(cvb.x + 400, cvb.y + 10); await page.mouse.down(); await page.mouse.move(cvb.x + 700, cvb.y + 10, { steps: 4 }); await page.mouse.up();
  w2 = await page.evaluate(() => ({ v: document.querySelector('#range-select').value, l: document.querySelector('#range-label').textContent }));
  ok(w2.v === 'sel' && /selection/.test(w2.l), `ruler drag -> ${w2.l}`);
  await page.keyboard.press('Escape');
  ok((await page.$eval('#range-select', s => s.value)) === 'all', 'Escape clears the selection');

  // W7: the warnings sample shows three banner items; the plain sample none
  const warn = names.find(n => /warnings/.test(n));
  await use(warn);
  let w7 = await page.evaluate(() => ({ hidden: document.querySelector('#warn').hidden, n: document.querySelectorAll('#warn > span').length, j: window.__nntr.metricsJSON().warnings }));
  ok(!w7.hidden && w7.n === 3, `warnings banner lists ${w7.n} items on the warnings sample`);
  ok(w7.j.fallbacks > 0 && w7.j.dropped.dsp === 137 && w7.j.clock_violations === 2, `metricsJSON.warnings: fallbacks ${w7.j.fallbacks}, dropped dsp ${w7.j.dropped.dsp}, clock ${w7.j.clock_violations}`);
  await page.click('#warn a'); await page.waitForTimeout(50);
  w7 = await page.evaluate(() => ({ f: document.querySelector('#search').value, sel: document.querySelector('#pane').innerText.includes('htp_disabled') }));
  ok(w7.f === 'fallback' && w7.sel, 'banner "show" filters and selects the first fallback slice');
  await use(asBuilt);
  ok(await page.$eval('#warn', e => e.hidden), 'no banner on the as-built sample');

  // batch B: W5 pool tail, W6 TOPS, W4 transport fit, W8 tokens
  const mjB = await page.evaluate(() => window.__nntr.metricsJSON());
  const quant = mjB.pool_tail.find(g => /quant f32->u8 AH/.test(g.kind));
  ok(quant && quant.few_units === true && quant.worst_tail > 1.1, `W5 pool tail flags the quant pool (min units ${quant && quant.min_units} < rule, worst tail ${quant && quant.worst_tail.toFixed(2)})`);
  const bigK = mjB.pool_tail.filter(g => !g.few_units).length;
  ok(mjB.pool_tail.length > 1, `W5 ${mjB.pool_tail.length} pool kinds, ${bigK} within the units rule`);
  const hmxPre = mjB.kernels.find(k => /qkv_proj: micro-mm x32/.test(k.name)), hmxDec = mjB.kernels.find(k => /micro-mm \(M=2/.test(k.name));
  ok(hmxPre && hmxDec && hmxPre.tops != null && hmxDec.tops != null && hmxPre.tops / hmxDec.tops > 5, `W6 prefill chunk TOPS ${hmxPre && hmxPre.tops.toFixed(3)} vs decode ${hmxDec && hmxDec.tops.toFixed(4)} (padding tax)`);
  ok(mjB.hmx_peak_tops === null, 'W6 peak TOPS stays null until measured');
  ok(mjB.transport.n > 50 && near(mjB.transport.fixed_us, 300, 45), `W4 transport fit: n ${mjB.transport.n}, fixed ${mjB.transport.fixed_us.toFixed(1)} µs (model 300), ${mjB.transport.us_per_mb.toFixed(1)} µs/MB (model 55)`);
  const tk = mjB.tokens.filter(t => t.phase === 'decode');
  const slope = (() => { const n = tk.length, mx = (n - 1) / 2, my = tk.reduce((s, t) => s + t.wall_us, 0) / n; return tk.reduce((s, t, i) => s + (i - mx) * (t.wall_us - my), 0) / tk.reduce((s, t, i) => s + (i - mx) ** 2, 0); })();
  ok(tk.length === 8 && slope > 0, `W8 ${tk.length} decode tokens, wall grows ${slope.toFixed(1)} µs/token (kv growth over transport jitter)`);
  ok(mjB.token_summary.ttft_us > 0 && mjB.token_summary.calls_per_token > 5, `W8 TTFT ${(mjB.token_summary.ttft_us / 1000).toFixed(1)} ms, ${mjB.token_summary.calls_per_token} calls/token`);
  for (const id of ['xport', 'tok']) { await page.evaluate(id => window.__nntr.tab(id), id); ok((await page.$eval('#pane svg', e => e.tagName)) === 'svg', `tab ${id} draws its chart`); }
  await page.evaluate(() => window.__nntr.tab('tok'));
  await page.click('#tok-table tr.click[data-i="3"]'); // phases[3] = decode token 3
  const v8 = await page.evaluate(() => { const v = window.__nntr.view(), t = window.__nntr.model().phases[3]; return Math.abs(v.t0 - t.ts) < 1e-6 && Math.abs(v.t1 - t.ts - t.dur) < 1e-6; });
  ok(v8, 'W8 clicking a token row zooms to it');
  await page.evaluate(() => window.__nntr.tab('ker'));
  ok((await page.$$eval('#pane button', bs => bs.map(b => b.textContent))).join() === 'Copy,Download', 'W12 export buttons on the Kernels tab');

  // W10: search navigation, thread/process collapsing, minimap
  await page.evaluate(() => window.__nntr.setFilter('softmax'));
  const hits = [];
  for (let i = 0; i < 3; i++) { await page.evaluate(() => window.__nntr.gotoMatch(1)); hits.push(await page.evaluate(() => { const e = window.__nntr.selected(); return [e.ts, e._label, e.tid]; })); }
  ok(hits.every(h => /softmax/.test(h[1])) && hits[0][0] <= hits[1][0] && hits[1][0] <= hits[2][0] && new Set(hits.map(h => h[0] + ':' + h[2])).size === 3, `W10 Enter x3 walks distinct softmax matches in ts order (${hits.map(h => h[0].toFixed(0) + '@' + h[2]).join(' ≤ ')})`);
  ok(/3 \/ \d+/.test(await page.$eval('#search-count', e => e.textContent)), 'W10 search counter shows k / n: ' + await page.$eval('#search-count', e => e.textContent));
  await page.evaluate(() => window.__nntr.fitMatch());
  const fitOk = await page.evaluate(() => { const e = window.__nntr.selected(), v = window.__nntr.view(); return v.t0 < e.ts && v.t1 > e.ts + e.dur && (v.t1 - v.t0) < e.dur * 1.5; });
  ok(fitOk, 'W10 f fits the view to the current match');
  const rowsBefore = await page.evaluate(() => window.__nntr.model().rows.length);
  await page.evaluate(() => { window.__nntr.collapsed.add('0:2'); window.__nntr.relayout(); });
  const rowsAfter = await page.evaluate(() => window.__nntr.model().rows.length);
  await page.evaluate(() => { window.__nntr.collapsed.delete('0:2'); window.__nntr.relayout(); });
  ok(rowsAfter < rowsBefore && (await page.evaluate(() => window.__nntr.model().rows.length)) === rowsBefore, `W10 collapsing HTP hides its rows (${rowsBefore} -> ${rowsAfter}) and restores them`);
  ok(await page.$eval('#mm', c => c.width > 100 && c.height > 10), 'W10 minimap canvas is drawn');
  await page.evaluate(() => window.__nntr.setFilter(''));

  // W11: state round-trips through the URL hash
  await page.evaluate(() => { window.__nntr.setView(20000, 30000); window.__nntr.tab('ker'); window.__nntr.setFilter('dequant'); });
  await page.waitForTimeout(350);
  const hash = await page.evaluate(() => location.hash);
  ok(/v=20000%2C30000|v=20000,30000/.test(hash) && /tab=ker/.test(hash) && /q=dequant/.test(hash), 'W11 hash carries view, tab and filter: ' + hash);
  await page.goto('file://' + path.resolve(page_path) + hash);
  await page.waitForFunction(() => window.__nntr && window.__nntr.model());
  const back = await page.evaluate(() => ({ v: window.__nntr.view(), tab: document.querySelector('#tabs button[aria-selected="true"]').dataset.tab, q: document.querySelector('#search').value }));
  ok(Math.abs(back.v.t0 - 20000) < 1 && Math.abs(back.v.t1 - 30000) < 1 && back.tab === 'ker' && back.q === 'dequant', `W11 reload restores view ${back.v.t0}-${back.v.t1}, tab ${back.tab}, filter ${back.q}`);
  await page.goto('file://' + path.resolve(page_path));
  await page.waitForFunction(() => window.__nntr && window.__nntr.model());

  // W3: A/B compare (as built vs pipelined)
  await use(asBuilt);
  await page.selectOption('#cmp-select', pipelined); await page.waitForTimeout(80);
  let w3 = await page.evaluate(() => { const T = window.__nntr.model(); return { n: T.models.length, names: T.models.map(m => m.name), rows: T.rows.length, tabs: [...document.querySelectorAll('#tabs button')].map(b => b.dataset.tab) }; });
  ok(w3.n === 2 && /^A: /.test(w3.names[0]) && /^B: /.test(w3.names[1]) && w3.tabs.includes('cmp'), `W3 compare loads B (${w3.names.join(' | ')}), Compare tab present`);
  await page.evaluate(() => window.__nntr.tab('cmp'));
  const cmpText = await page.$eval('#pane', e => e.innerText);
  const compLine = cmpText.split('\n').find(l => /parallel compression/.test(l)) || '';
  ok(/1\.17×\s+1\.45×\s+\+0\.2[0-9]/.test(compLine.replace(/\t/g, ' ')), 'W3 compression A 1.17× -> B 1.45×: ' + compLine.trim());
  const ovl = cmpText.split('\n').find(l => /HMX ∥ HVX/.test(l)) || '';
  ok(/HMX ∥ HVX\s+0\.000 µs\s+[1-9]/.test(ovl.replace(/\t/g, ' ')), 'W3 overlap 0 -> >0: ' + ovl.trim());
  const deq = await page.$$eval('#cmp-kernels tr', trs => trs.map(t => t.innerText).find(t => /qkv_proj: dequant i32->f32/.test(t)) || '');
  ok(/\t0\.0%\t/.test(deq) || /0\.0%/.test(deq.split('\t')[9] || ''), 'W3 qkv dequant mean delta 0.0%: ' + deq.replace(/\t/g, ' | '));
  await page.selectOption('#view-select', 'a'); await page.waitForTimeout(50);
  const rowsA = await page.evaluate(() => window.__nntr.model().rows.length);
  ok(rowsA < w3.rows, `W3 View: A hides B's rows (${w3.rows} -> ${rowsA})`);
  await page.selectOption('#cmp-select', ''); await page.waitForTimeout(50);
  ok((await page.evaluate(() => window.__nntr.model().models.length)) === 1, 'W3 Compare: none restores a single model');

  // W9: QNN optrace fixture converted and shown as pid 3 beside ours
  const qnn = names.find(n => /^qnn/.test(n));
  await page.selectOption('#cmp-select', qnn); await page.waitForTimeout(80);
  const w9 = await page.evaluate(() => { const B = window.__nntr.model().models[1], p = B.procs[0]; return { pid: p.pid, tids: [...p.threads.keys()].sort((a, b) => a - b), n: B.all.length, hmx: B.all.filter(e => e.cat === 'dsp.hmx').length, cls: B.all.map(e => e.args.class), cyc: B.all.every(e => e.args.cycles > 0), softmaxCpe: (() => { const k = window.__nntr.kernels({ t0: 0, t1: B.tmax }); return null; })() }; });
  ok(w9.pid === 3 && w9.tids.join() === '256,512,513,768' && w9.n === 7 && w9.hmx === 2, `W9 converted optrace: pid ${w9.pid}, tids ${w9.tids.join()}, ${w9.n} slices (views dropped), ${w9.hmx} HMX`);
  ok(w9.cyc && w9.cls.includes('matmul') && w9.cls.includes('softmax') && w9.cls.includes('transpose') && w9.cls.includes('dma'), `W9 classes and cycles carried over: ${[...new Set(w9.cls)].join(',')}`);
  const qref = await page.evaluate(() => { const B = window.__nntr.model().models[1]; return window.__nntr.kernelsOf(B).map(k => [k.name, k.cls, +k.cpe.toFixed(2)]); });
  const sm = qref.find(k => /Softmax/.test(k[0])), tr = qref.find(k => /Transpose/.test(k[0]));
  ok(sm && sm[1] === 'softmax' && near(sm[2], 0.24, 0.01) && tr && near(tr[2], 0.71, 0.01), `W9 cy/elem from the fixture's cycles: softmax ${sm && sm[2]} (ref 0.24), transpose ${tr && tr[2]} (ref 0.71)`);
  await page.selectOption('#cmp-select', ''); await page.waitForTimeout(50);

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

  // dump metrics JSON for the python cross-check (W12): whole-trace range of the as-built sample
  await use(asBuilt); await page.selectOption('#range-select', 'all'); await page.waitForTimeout(50);
  const mj = await page.evaluate(() => window.__nntr.metricsJSON());
  fs.mkdirSync(fixtures, { recursive: true });
  fs.writeFileSync(path.join(fixtures, 'viewer_metrics.json'), JSON.stringify(mj, null, 1));

  ok(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' / ') : ''));
  await page.screenshot({ path: path.join(fixtures, 'viewer.png') });
  await browser.close();
  console.log(failures ? `${failures} check(s) FAILED` : 'all checks passed');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
