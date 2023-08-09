from pyppeteer import launch

async def scrape():
    urls = []
    names = []
    browser = await launch({"headless": True})  
    page = await browser.newPage()

    await page.goto('https://africaopendata.org/dataset/sensorsafrica-airquality-archive-nairobi') # Load OpenAfrica Web pag
    downloadUrl = await page.querySelectorAll("a.resource-url-analytics")   # Select web element with link
    titles = await page.querySelectorAll("a.heading")   # Select web element with file name

    for i in range(len(downloadUrl)):
        urls.append(await page.evaluate("(element) => element.getAttribute('href')",downloadUrl[i]))    # Get download link
        names.append(await page.evaluate("(element) => element.getAttribute('title')",titles[i]))   # Get file name
        names[-1] = names[-1].lower().replace(" ","_")  # format file name

    await browser.close()
    return urls,names


