// Screenshots each figure page at 2x (3840x2160 PNG) once its web font has
// loaded: node render.js
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 },
                                       deviceScaleFactor: 2 });
  for (const n of ['fsu_1_placement', 'fsu_2_prefill_prefetch', 'fsu_3_decode_cache',
                  'fsu_a_overview', 'fsu_b_prefill_before_after', 'fsu_c_decode_hits',
                  'fsu_prefill_prefetch', 'fsu_expert_memory_table']) {
    await page.goto('file://' + path.join(__dirname, n + '.html'));
    await page.evaluate(() => document.fonts.ready);
    const ok = await page.evaluate(() => document.fonts.check('700 20px "Noto Sans KR"'));
    if (!ok) throw new Error(n + ': Noto Sans KR did not load');
    await page.screenshot({ path: path.join(__dirname, n + '.png') });
    console.log('rendered', n);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
