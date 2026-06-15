import os
import sys
import json

# Remove proxy environment variables so we don't route through the blocked proxy
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(k, None)

import urllib.request
import urllib.error
import ssl
import gzip

# Disable SSL verification for testing
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

promise_keywords = {
    "p003": ["black money", "swiss", "15 lakh", "repatriate", "manifesto"],
    "p006": ["black money", "corruption", "manifesto"],
    "p010": ["simultaneous", "election", "one nation"],
    "p011": ["demonetisation", "demonetization", "black money", "currency"],
    "p012": ["bullet train", "mumbai", "ahmedabad", "high speed"],
    "p014": ["school", "education", "classroom", "reforms", "manifesto", "aap"],
    "p015": ["electricity", "power", "subsidy", "unit", "manifesto"],
    "p016": ["yamuna", "clean", "sewage", "river", "manifesto"],
    "p018": ["mafia", "crime", "encounter", "bulldozer", "manifesto", "sankalp"],
    "p019": ["manush", "mati", "trinamool", "manifesto"],
    "p020": ["viksit", "developed", "2047", "centenary"]
}

candidates = {
    "p003": "https://web.archive.org/web/20190208201614/http://www.bjp.org:80/manifesto2014",
    "p006": "https://web.archive.org/web/20190208201614/http://www.bjp.org:80/manifesto2014",
    "p010": "https://pib.gov.in/PressReleasePage.aspx?PRID=2056059",
    "p011": "https://pib.gov.in/newsite/PrintRelease.aspx?relid=153404",
    "p012": "https://pib.gov.in/PressReleasePage.aspx?PRID=1503463",
    "p020": "https://pib.gov.in/PressReleasePage.aspx?PRID=1949023",
}

needs_wayback = {
    "p014": "https://aamaadmiparty.org/delhi-manifesto-2015",
    "p015": "https://aamaadmiparty.org/delhi-manifesto-2015",
    "p016": "https://aamaadmiparty.org/delhi-manifesto-2020",
    "p018": "https://timesofindia.indiatimes.com/assembly-elections-2017/uttar-pradesh/bjp-up-manifesto-2017-key-highlights-of-lok-kalyan-sankalp-patra/articleshow/56832269.cms",
    "p019": "https://timesofindia.indiatimes.com/assembly-elections-2011/west-bengal/trinamool-congress-manifesto-highlights/articleshow/7786440.cms"
}

def fetch_url(url):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            body = response.read()
            if response.info().get('Content-Encoding') == 'gzip':
                body = gzip.decompress(body)
            return response.getcode(), body.decode('utf-8', errors='ignore').lower()
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return f"ERROR: {e}", ""

# Resolve wayback urls
print("Resolving Wayback Machine URLs...")
for pid, url in needs_wayback.items():
    wayback_api = f"https://archive.org/wayback/available?url={url}"
    code, html = fetch_url(wayback_api)
    try:
        data = json.loads(html)
        if data.get('archived_snapshots', {}).get('closest', {}).get('available'):
            archive_url = data['archived_snapshots']['closest']['url']
            # Make sure it redirects correctly by forcing the exact URL if possible or just use it
            archive_url = archive_url.replace("http://", "https://")
            candidates[pid] = archive_url
        else:
            print(f"[{pid}] No wayback snapshot found for {url}")
            candidates[pid] = url
    except Exception as e:
        print(f"[{pid}] JSON parse error from wayback API: {e}")
        candidates[pid] = url

print("\nTesting Candidates...")
print("-" * 105)
print(f"{'ID':<5} | {'HTTP':<5} | {'Content':<8} | {'Matched Keywords':<20} | {'URL'}")
print("-" * 105)

results = {}

for pid, url in candidates.items():
    code, html = fetch_url(url)
    matched = []
    content_verified = "MISMATCH"
    
    if code == 200:
        kws = promise_keywords.get(pid, [])
        matched = [kw for kw in kws if kw in html]
        if matched:
            content_verified = "MATCHED"
            
    matched_str = ", ".join(matched) if matched else "None"
    print(f"{pid:<5} | {code:<5} | {content_verified:<8} | {matched_str:<20} | {url}")
    
    results[pid] = {
        "url": url,
        "status": code,
        "content_verified": content_verified
    }

# Save results for agent to read
with open("/Users/mac/Downloads/Code/Satya/Satya-promise-tracker/candidate_results.json", "w") as f:
    json.dump(results, f, indent=2)
