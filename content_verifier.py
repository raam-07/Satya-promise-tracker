import json
import urllib.request
import urllib.error
import os
import ssl
import gzip

PROMISES_PATH = os.path.join(os.path.dirname(__file__), 'promises.json')

with open(PROMISES_PATH) as f:
    data = json.load(f)

# Define keywords to search in content to verify it's the correct page
promise_keywords = {
    "p001": ["farmer", "income", "2022"],
    "p002": ["manifesto", "job", "employment"],
    "p003": ["black money", "swiss", "15 lakh", "repatriate"],
    "p004": ["awas", "housing", "pucca", "rural", "urban"],
    "p005": ["5 trillion", "five trillion", "gdp", "growth"],
    "p006": ["black money", "corruption", "manifesto"],
    "p007": ["electoral", "bond", "transparency", "funding"],
    "p008": ["nal", "jal", "water", "household", "jeevan"],
    "p009": ["civil code", "ucc", "manifesto"],
    "p010": ["simultaneous", "election", "one nation"],
    "p011": ["demonetisation", "demonetization", "black money", "currency"],
    "p012": ["bullet train", "mumbai", "ahmedabad", "high speed"],
    "p013": ["rohingya", "illegal", "border", "migrant"],
    "p014": ["school", "education", "classroom", "reforms"],
    "p015": ["electricity", "power", "subsidy", "unit"],
    "p016": ["yamuna", "clean", "sewage", "river"],
    "p017": ["nyay", "minimum income", "poor", "manifesto"],
    "p018": ["mafia", "crime", "encounter", "bulldozer"],
    "p019": ["manush", "mati", "trinamool", "manifesto"],
    "p020": ["viksit", "developed", "2047", "centenary"]
}

# Full browser headers to bypass CDN blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Bypass SSL errors (common on government websites)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print("Checking both URL status & verifying page content keywords...")
print("-" * 105)
print(f"{'ID':<5} | {'HTTP':<5} | {'Content':<8} | {'Matched Keywords':<20} | {'URL'}")
print("-" * 105)

for p in data['promises']:
    pid = p['id']
    url = p.get('source_url', '').strip()
    if not url:
        continue

    req = urllib.request.Request(url, headers=HEADERS)
    status_code = "???"
    content_verified = "No"
    matched = []
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as response:
            status_code = response.getcode()
            
            # Read and handle gzip decompression
            body = response.read()
            if response.info().get('Content-Encoding') == 'gzip':
                body = gzip.decompress(body)
            html_content = body.decode('utf-8', errors='ignore').lower()
            
            # Check keywords
            kws = promise_keywords.get(pid, [])
            matched = [kw for kw in kws if kw in html_content]
            if matched:
                content_verified = "MATCHED"
            else:
                content_verified = "MISMATCH"
                
    except urllib.error.HTTPError as e:
        status_code = e.code
    except Exception as e:
        status_code = "ERROR"

    matched_str = ", ".join(matched) if matched else "None"
    print(f"{pid:<5} | {status_code:<5} | {content_verified:<8} | {matched_str:<20} | {url}")
