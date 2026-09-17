"""Offline browser checks for the frozen bilateral study supplement."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = Path('/tmp/novigrad-bilateral-report-qa')


def main():
    OUT.mkdir(exist_ok=True)
    checks = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            headless=True)
        for name in ('bilateral.html', 'bilateral.ko.html'):
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors, remote = [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: remote.append(request.url)
                    if request.url.startswith(('http:', 'https:')) else None)
            page.goto((ROOT / name).as_uri())
            assert page.locator('svg').count() >= 1
            assert page.locator('section').count() >= 9
            assert page.locator('table').count() >= 6
            assert page.locator('details').count() >= 8
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(OUT / f'{name}.desktop.png'))
            disclosure = page.locator('details').first
            was_open = disclosure.get_attribute('open') is not None
            disclosure.locator('summary').click()
            assert (disclosure.get_attribute('open') is not None) != was_open
            if was_open:
                disclosure.locator('summary').click()
            assert disclosure.get_attribute('open') is not None
            assert disclosure.locator('article.failure pre').count() > 0
            page.set_viewport_size({'width': 390, 'height': 844})
            page.evaluate('scrollTo(0,0)')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(OUT / f'{name}.mobile.png'))
            page.evaluate("document.querySelectorAll('details').forEach(e => e.open=true)")
            page.emulate_media(media='print')
            page.set_viewport_size({'width': 794, 'height': 1123})
            page.evaluate('scrollTo(0,0)')
            page.screenshot(path=str(OUT / f'{name}.print.png'))
            assert not errors and not remote
            checks[name] = {
                'sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                'offline': True, 'desktop_mobile_no_page_overflow': True,
                'failure_disclosures_work': True, 'no_browser_errors': True,
                'sections': page.locator('section').count(),
                'print_media_checked_with_details_open': True}
            page.close()
        browser.close()
    (ROOT / 'bilateral-browser-verification.json').write_text(
        json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
