from playwright.sync_api import sync_playwright
import time

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto('http://localhost:5173')
    time.sleep(3) # Wait for animations/load
    page.screenshot(path='frontend_screenshot.png', full_page=True)
    browser.close()
    print("Screenshot saved.")

with sync_playwright() as playwright:
    run(playwright)
