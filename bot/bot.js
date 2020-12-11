
const puppeteer = require('puppeteer');
const fs = require('fs');
const csv = require('neat-csv');
const {installMouseHelper} = require('./install-mouse-helper');

const csvfile = 'example_trajectory.csv'
// 'user_data/training_files/user12/session_2144641057'
const getcsv = async (filename) => {
    const f = await fs.promises.readFile(filename)
    return csv(f) 
}

(async () => {
    const browser = await puppeteer.launch({
        headless: false
    });
    const page = await browser.newPage();

    // set the viewport so we know the dimensions of the screen
    await page.setViewport({ width: 1920, height: 1080 })
    await installMouseHelper(page);
    // go to a page setup for mouse event tracking
    await page.goto('http://unixpapa.com/js/testmouse.html')

    // click an area
    await page.mouse.click(132, 103, { button: 'left' })

    // the screenshot should show feedback from the page that right part was clicked.
    await page.screenshot({ path: 'mouse_click.png' })
    
    const results = await getcsv(csvfile)

    for (const row of results) {
        try {
            await page.mouse.move(1920 *parseFloat(row['x']),1080 *parseFloat(row['y']))
        }
        catch (error) {
            if (!page.browser().isConnected()) return
            console.log('Warning: could not move mouse, error message:', error)
        }
    }
    await browser.close()

})()
