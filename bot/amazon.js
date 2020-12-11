const puppeteer = require('puppeteer')
const {installMouseHelper} = require('./install-mouse-helper');

const url = 'https://www.amazon.com/gp/product/B001C139JU?pf_rd_r=9P965PP3BJ5DX4AS31F6&pf_rd_p=9d9090dd-8b99-4ac3-b4a9-90a1db2ef53b'
try {
  (async () => {
    const browser = await puppeteer.launch({ headless: false})
    const page = await browser.newPage()
    await installMouseHelper(page);
    await page.setViewport({ width: 1280, height: 800 })
    await page.goto(url, {waitUntil: 'networkidle2'})
    await page.click('#add-to-cart-button');
    await page.waitForSelector('huc-v2-order-row-confirm-text')
    await browser.close()
  })()
} catch (err) {
  console.error(err)
}
