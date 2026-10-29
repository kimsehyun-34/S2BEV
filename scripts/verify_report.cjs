// Optional browser smoke check. Requires an existing playwright-core installation.
// node scripts/verify_report.cjs [path-to-playwright-core]
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const fs = require('node:fs');
const { chromium } = require(process.argv[2] || 'playwright-core');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || undefined });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(path.resolve('reports/index.html')).href);
    assert.equal(await page.locator('#team option').count(), 20);
    assert.equal(await page.locator('#team-title').textContent(), 'Arsenal');
    assert.equal(await page.locator('#formation').inputValue(), '4-2-3-1');
    assert.equal(await page.locator('#points').textContent(), '1.702 예상 승점 / 경기');
    assert.equal(await page.locator('#test-accuracy').textContent(), '52.5%');
    for (let index = 0; index < 20; index++) {
      await page.selectOption('#team', String(index));
      const formations = await page.locator('#formation option').evaluateAll(nodes => nodes.map(n => n.value));
      for (const formation of formations) {
        await page.selectOption('#formation', formation);
        assert.equal(await page.locator('#pitch circle[fill="#65e3b2"], #pitch circle[fill="#edbd68"]').count(), 11);
        assert.match(await page.locator('#points').textContent(), /^\d\.\d{3}/);
      }
    }
    await page.locator('#teams tr[data-index="1"]').press('Enter');
    assert.equal(await page.locator('#team-title').textContent(), 'Arsenal');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: 'reports/explorer-preview.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: 'reports/explorer-preview-mobile.png', fullPage: true });
    for (const href of await page.locator('a').evaluateAll(nodes => nodes.map(n => n.getAttribute('href')))) {
      if (!href.startsWith('https://')) assert.ok(fs.existsSync(path.join('reports', href)), href);
    }
    assert.equal(fs.readFileSync('reports/index.html', 'utf8'), fs.readFileSync('reports/explorer.html', 'utf8'));
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto(pathToFileURL(path.resolve('reports/analysis.html')).href);
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
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.goto(pathToFileURL(path.resolve('reports/focused/index.html')).href);
    const focused = JSON.parse(fs.readFileSync('reports/focused/metrics.json', 'utf8'));
    assert.equal(await page.locator('#games').textContent(), focused.data.unique_matches.toLocaleString());
    assert.equal(await page.locator('#coverage tr').count(), 6);
    assert.equal(await page.locator('#metrics tr').count(), 9);
    assert.equal(await page.locator('#search tr').count(), 8);
    assert.ok(await page.locator('img').first().evaluate(img => img.complete && img.naturalWidth > 0));
    for (const club of ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham Hotspur']) {
      await page.selectOption('#club', club);
      assert.ok((await page.locator('#club-summary').textContent()).includes('경기 수집'));
      assert.equal(await page.locator('#club-metrics tr').count(), 8);
    }
    for (const href of await page.locator('a').evaluateAll(nodes => nodes.map(n => n.getAttribute('href')))) {
      if (!href.startsWith('https://')) assert.ok(fs.existsSync(path.join('reports/focused', href)), href);
    }
    await page.screenshot({ path: 'reports/focused/preview.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: 'reports/focused/preview-mobile.png', fullPage: true });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ explorerTeams: 20, formationsChecked: 260, withheldRecommendations: 20, focusedClubs: 6, consoleErrors: errors, mobileOverflow: false, screenshots: 6 }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
