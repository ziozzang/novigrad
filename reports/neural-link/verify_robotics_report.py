"""Check offline rendering, citations, disclosures and mobile layout."""
from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = Path('/tmp/novigrad-robotics-report-qa')


def main():
    OUT.mkdir(exist_ok=True)
    checks = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
        for name in ('robotics.html', 'robotics.ko.html'):
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors, remote = [], []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('request', lambda r: remote.append(r.url) if r.url.startswith(('http:', 'https:')) else None)
            page.goto((ROOT / name).as_uri())
            assert page.locator('svg').count() == 1
            assert page.locator('tbody tr').count() == 10
            assert page.locator('h2').count() >= 10
            assert page.locator('a[href^="https:"]').count() >= 15
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(OUT / (name + '.desktop.png')))
            page.locator('details summary').click()
            assert page.locator('details').get_attribute('open') is not None
            page.set_viewport_size({'width': 390, 'height': 844})
            page.evaluate('scrollTo(0,0)')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert page.locator('.table-scroll').evaluate('(e) => e.scrollWidth > e.clientWidth')
            page.screenshot(path=str(OUT / (name + '.mobile.png')))
            page.emulate_media(media='print')
            page.set_viewport_size({'width': 794, 'height': 1123})
            page.screenshot(path=str(OUT / (name + '.print.png')))
            assert not errors and not remote
            checks[name] = {'sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                            'offline': True, 'desktop_mobile_no_overflow': True,
                            'mobile_table_scrollable': True, 'details_work': True,
                            'print_media_checked': True, 'no_browser_errors': True}
            page.close()
        browser.close()
    (ROOT / 'robotics-browser-verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
