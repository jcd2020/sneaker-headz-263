

const puppeteer = require('puppeteer')
const {installMouseHelper} = require('./install-mouse-helper');
const url = 'https://www.ssense.com/en-us/men/product/neil-barrett/off-white-li-ning-edition-essence-2.3-sneakers/7118271'
try {
  (async () => {
    const browser = await puppeteer.launch({ headless: false})
    const page = await browser.newPage()
    await installMouseHelper(page);
    await page.setViewport({ width: 1280, height: 800 })
    await page.goto(url, { waitUntil: 'networkidle2' })
    await page.select('#size', '8.5_202368M23710102')
    await page.click('button.button--clear-styles.s-button')
  })()
} catch (err) {
  console.error(err)
}
