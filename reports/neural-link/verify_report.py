"""Browser QA for the offline bilingual report. Requires Playwright and Chrome."""
import argparse
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--browser',default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');p.add_argument('--screenshots',type=Path,default=Path('/tmp/novigrad-report-qa'));args=p.parse_args();args.screenshots.mkdir(parents=True,exist_ok=True)
 expected=json.loads((ROOT/'report-data.json').read_text());results={}
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path=args.browser,headless=True)
  for filename in ['report.html','report.ko.html']:
   page=browser.new_page(viewport={'width':1440,'height':1080},accept_downloads=True)
   errors=[];remote=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   page.on('request',lambda r:remote.append(r.url) if r.url.startswith(('http://','https://')) else None)
   page.goto((ROOT/filename).as_uri());page.wait_for_load_state('load')
   assert not errors,errors
   assert page.locator('section').count()==15
   assert page.locator('figure svg[role="img"]').count()==4
   assert page.locator('#case-rows tr').count()==64
   assert page.locator('#report-data').evaluate('(e)=>JSON.parse(e.textContent)')==expected
   structure=page.evaluate('''()=>{const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return {duplicateIds:ids.filter((x,i)=>ids.indexOf(x)!==i),brokenAnchors:[...document.querySelectorAll('a[href^="#"]')].map(e=>e.hash.slice(1)).filter(x=>!document.getElementById(x)),externalAssets:[...document.querySelectorAll('script[src],link[rel="stylesheet"],img[src]')].map(e=>e.src||e.href)}}''')
   assert structure=={'duplicateIds':[],'brokenAnchors':[],'externalAssets':[]},structure
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.screenshot(path=str(args.screenshots/(filename+'.desktop.png')))
   page.locator('#fig-1').scroll_into_view_if_needed();page.locator('#fig-1').screenshot(path=str(args.screenshots/(filename+'.figure.png')))
   page.select_option('#language','ko');page.select_option('#outcome','error')
   assert page.locator('#case-rows tr:not([hidden])').count()==4
   page.select_option('#mode','fixed')
   assert page.locator('#case-rows tr:not([hidden])').count()==16
   page.fill('#query','zzzz-impossible-text');assert page.locator('#case-rows tr:not([hidden])').count()==0
   page.fill('#query','');page.select_option('#language','all');page.select_option('#outcome','all');page.select_option('#mode','pca')
   with page.expect_download() as task:page.click('#download-data')
   downloaded=task.value;path=args.screenshots/(filename+'.download.json');downloaded.save_as(path)
   assert json.loads(path.read_text())==expected
   page.set_viewport_size({'width':390,'height':844});page.evaluate('scrollTo(0,0)')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.screenshot(path=str(args.screenshots/(filename+'.mobile.png')))
   # CSS print must hide controls and make even filtered-out case rows visible.
   page.select_option('#outcome','error');page.emulate_media(media='print')
   assert page.locator('.sidebar').evaluate('(e)=>getComputedStyle(e).display')=='none'
   assert page.locator('#case-rows tr').evaluate_all('(rows)=>rows.every(e=>getComputedStyle(e).display!=="none")')
   page.set_viewport_size({'width':794,'height':1123});page.evaluate('scrollTo(0,0)');page.screenshot(path=str(args.screenshots/(filename+'.print.png')))
   assert not remote,remote
   assert not errors,errors
   results[filename]={'sha256':sha(ROOT/filename),'sections':15,'figures':4,'case_rows':64,'offline_no_external_requests':True,'unique_ids_and_valid_anchors':True,'desktop_no_page_overflow':True,'mobile390_no_page_overflow':True,'filter_expected_counts':[4,16,0],'json_download_matches':True,'print_controls_hidden_all_cases_visible':True,'javascript_errors':errors}
   page.close()
  browser.close()
 report={'browser_executable':args.browser,'checks':results,'scope':'Static report data and browser interactions; no model retraining. Print media checked; pagination review remains journal-specific.'}
 (ROOT/'browser-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
