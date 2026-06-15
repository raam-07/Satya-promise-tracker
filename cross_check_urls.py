import json
import subprocess
import os

# Path to the local promises.json
PROMISES_PATH = os.path.join(os.path.dirname(__file__), 'promises.json')

with open(PROMISES_PATH) as f:
    data = json.load(f)

urls = set()
for p in data['promises']:
    if p.get('source_url'):
        urls.add((p['id'], 'source', p['source_url']))
    for a in p.get('evidence_articles', []):
        if a.get('url'):
            urls.add((p['id'], 'evidence', a['url']))

print(f"Loaded {len(urls)} unique URLs to check.")
print(f"{'Promise ID':<12} | {'Type':<8} | {'HTTP':<5} | {'URL'}")
print("-" * 80)

# Create a clean environment without proxy variables to check directly
curl_env = os.environ.copy()
curl_env.pop('HTTP_PROXY', None)
curl_env.pop('HTTPS_PROXY', None)
curl_env.pop('http_proxy', None)
curl_env.pop('https_proxy', None)

for pid, utype, url in sorted(urls):
    try:
        # Run curl to get the HTTP status code
        res = subprocess.run(
            ['curl', '-s', '-I', '-L', '-o', '/dev/null', '-w', '%{http_code}', '--connect-timeout', '10', url],
            capture_output=True, text=True, env=curl_env
        )
        status_code = res.stdout.strip()
        print(f"{pid:<12} | {utype:<8} | {status_code:<5} | {url}")
    except Exception as e:
        print(f"{pid:<12} | {utype:<8} | ERROR | {url} ({e})")
