// Browser smoke check for reports/focused/lab.html. Requires an existing playwright-core installation.
// CHROME_PATH=<chrome.exe> node scripts/verify_lab.cjs [path-to-playwright-core] [screenshot-dir]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.argv[2] || 'playwright-core');

const shotDir = process.argv[3];
const STEPS = ['s1', 's2', 's3', 's4'];

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || undefined });
  const summary = { steps: {}, errors: [] };
  try {
    for (const [width, theme] of [[1280, 'light'], [1280, 'dark'], [390, 'light']]) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: theme });
      const page = await context.newPage();
      page.on('pageerror', (e) => summary.errors.push(`${width}/${theme} pageerror: ${e.message}`));
      page.on('console', (m) => { if (m.type() === 'error') summary.errors.push(`${width}/${theme} console: ${m.text()}`); });
      await page.goto(pathToFileURL(path.resolve('reports/focused/lab.html')).href);
      assert.equal(await page.locator('[role=tab]').count(), 4);
      for (const id of STEPS) {
        await page.locator(`#tab-${id}`).click();
        await page.waitForTimeout(400);
        const panel = page.locator(`#${id}`);
        assert.ok(await panel.isVisible(), `${id} visible`);
        assert.ok((await panel.locator('.takeaway').count()) >= 1, `${id} has a takeaway`);
        assert.ok((await panel.locator('.card').count()) >= 3, `${id} has content cards`);
        assert.ok(!(await panel.textContent()).includes('준비 중'), `${id} is built`);
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
        assert.ok(!overflow, `${id} has no horizontal overflow at ${width}`);
        const blankCharts = await panel.locator('svg.chart-svg').evaluateAll((nodes) => nodes.filter((n) => n.childElementCount === 0 && n.getClientRects().length > 0).length);
        assert.equal(blankCharts, 0, `${id} has empty charts`);
        summary.steps[`${id}@${width}/${theme}`] = { cards: await panel.locator('.card').count(), svgCharts: await panel.locator('svg.chart-svg').count(), canvases: await panel.locator('canvas').count() };
        if (shotDir) { fs.mkdirSync(shotDir, { recursive: true }); await page.screenshot({ path: path.join(shotDir, `lab-${id}-${width}-${theme}.png`), fullPage: true }); }
      }
      for (const href of await page.locator('a').evaluateAll((nodes) => nodes.map((n) => n.getAttribute('href')))) {
        if (href && !href.startsWith('https://') && !href.startsWith('#')) assert.ok(fs.existsSync(path.join('reports/focused', href)), `missing link target ${href}`);
      }
      await context.close();
    }
    assert.deepEqual(summary.errors, []);
    console.log(JSON.stringify(summary));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
