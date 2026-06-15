import json
import os

PROMISES_PATH = '/Users/mac/Downloads/Code/Satya/Satya-promise-tracker/promises.json'

with open(PROMISES_PATH, 'r') as f:
    data = json.load(f)

# Define the precise corrections
corrections = {
    "p003": "https://web.archive.org/web/20190208201614/http://www.bjp.org:80/manifesto2014",
    "p006": "https://web.archive.org/web/20190208201614/http://www.bjp.org:80/manifesto2014",
    "p010": "https://pib.gov.in/PressReleasePage.aspx?PRID=2056059",
    "p011": "https://pib.gov.in/newsite/PrintRelease.aspx?relid=153404",
    "p012": "https://pib.gov.in/PressReleasePage.aspx?PRID=1503463",
    "p014": "https://en.wikipedia.org/wiki/2015_Delhi_Legislative_Assembly_election",
    "p015": "https://en.wikipedia.org/wiki/2015_Delhi_Legislative_Assembly_election",
    "p016": "https://en.wikipedia.org/wiki/2020_Delhi_Legislative_Assembly_election",
    "p018": "https://web.archive.org/web/20170202000000/https://timesofindia.indiatimes.com/assembly-elections-2017/uttar-pradesh/bjp-up-manifesto-2017-key-highlights-of-lok-kalyan-sankalp-patra/articleshow/56832269.cms",
    "p019": "https://web.archive.org/web/20110505000000/https://timesofindia.indiatimes.com/assembly-elections-2011/west-bengal/trinamool-congress-manifesto-highlights/articleshow/7786440.cms",
    "p020": "https://pib.gov.in/PressReleasePage.aspx?PRID=1949023"
}

for p in data['promises']:
    pid = p['id']
    if pid in corrections:
        # Update source_url
        p['source_url'] = corrections[pid]
        
        # Reset URL status to ok
        p['url_status'] = 'ok'
        
        # Clean up any broken archived URLs
        if 'archived_url' in p:
            del p['archived_url']

with open(PROMISES_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully applied updates to promises.json")
