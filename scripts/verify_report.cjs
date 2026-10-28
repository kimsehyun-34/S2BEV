// Optional browser smoke check. Requires an existing playwright-core installation.
// node scripts/verify_report.cjs [path-to-playwright-core]
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const fs = require('node:fs');
const { chromium } = require(process.argv[2] || 'playwright-core');

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(path.resolve('reports/index.html')).href);
    const metrics = JSON.parse(fs.readFileSync('reports/benchmark_metrics.json', 'utf8'));
    const deployed = metrics.deployment.uses_formation ? 'with_formation' : 'without_formation';
    assert.match(await page.locator('h1').textContent(), /최적 전술/);
    assert.equal(await page.getByText('최적 대형 판정 보류', { exact: true }).count(), 20);
    assert.equal(await page.locator('tbody tr').count(), 33);
    assert.ok((await page.locator('.stats').textContent()).includes((100 * metrics.test[deployed].accuracy).toFixed(2) + '%'));
    assert.ok(await page.locator('img').evaluate(img => img.complete && img.naturalWidth > 0));
    for (const href of await page.locator('a').evaluateAll(nodes => nodes.map(n => n.getAttribute('href')))) {
      if (!href.startsWith('https://')) assert.ok(fs.existsSync(path.join('reports', href)), href);
    }
    await page.screenshot({ path: 'reports/preview.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: 'reports/preview-mobile.png', fullPage: true });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ teams: 20, withheldRecommendations: 20, consoleErrors: errors, mobileOverflow: false, screenshots: 2 }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
