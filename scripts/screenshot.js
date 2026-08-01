const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  await page.goto(process.env.FRONTEND_URL || 'http://localhost:5173', { waitUntil: 'networkidle0' });
  await new Promise(resolve => setTimeout(resolve, 2000));
  await page.screenshot({ path: 'docs/frontend_screenshot.png', fullPage: true });
  await browser.close();
  console.log('Screenshot saved to frontend_screenshot.png');
})();
