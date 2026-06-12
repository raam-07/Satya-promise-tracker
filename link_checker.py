# ==============================================================================
# SATYA — PROMISE LINK CHECKER
#
# Verifies every source_url and evidence article URL in promises.json:
#   1. Checks liveness (HTTP status) with a real browser User-Agent
#   2. Dead links: looks up the Wayback Machine for an archived snapshot
#      and stores it as `archived_url` (the original URL is NEVER removed)
#   3. Live links without a snapshot: optionally asks the Wayback Machine
#      to save one NOW (set ARCHIVE_LIVE=true) so future rot is recoverable
#   4. Flags homepage-only sources (path-less URLs a reader can't verify)
#
# Fields written (additive only — never deletes anything):
#   url_status: "ok" | "dead"        url_checked_at: ISO date
#   archived_url: wayback snapshot   source_quality: "homepage_only"
#
# Runs weekly via GitHub Actions (link_checker.yml).
# ==============================================================================

import json
import logging
import os
import time
from datetime import datetime
from urllib.parse import urlparse, quote

import requests

PROMISES_PATH = './promises.json'
TIMEOUT = 15
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0 Safari/537.36 SatyaLinkChecker/1.0'
}
ARCHIVE_LIVE = os.environ.get('ARCHIVE_LIVE', '').lower() in ('1', 'true', 'yes')
# Be polite to servers and to the Wayback Machine
SLEEP_BETWEEN_CHECKS = 1.0
SLEEP_BETWEEN_SAVES = 8.0
MAX_SAVES_PER_RUN = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_url(url):
    """Returns 'ok' or 'dead'. Tries HEAD first, falls back to GET."""
    try:
        r = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code in (405, 403, 400):  # some servers reject HEAD
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True, stream=True)
            r.close()
        return 'ok' if r.status_code < 400 else 'dead'
    except requests.RequestException:
        return 'dead'


def wayback_lookup(url):
    """Returns the closest archived snapshot URL, or None."""
    try:
        r = requests.get(
            'https://archive.org/wayback/available',
            params={'url': url}, headers=HEADERS, timeout=TIMEOUT,
        )
        snap = r.json().get('archived_snapshots', {}).get('closest', {})
        if snap.get('available') and snap.get('url'):
            return snap['url'].replace('http://', 'https://', 1)
    except Exception as e:
        logging.warning(f"wayback lookup failed for {url}: {e}")
    return None


def wayback_save(url):
    """Best-effort 'Save Page Now'. Returns True if accepted."""
    try:
        r = requests.get(f'https://web.archive.org/save/{quote(url, safe=":/?&=%")}',
                         headers=HEADERS, timeout=60)
        return r.status_code < 400
    except Exception:
        return False


def process_entry(entry, url_key, stats, saves_done):
    url = (entry.get(url_key) or '').strip()
    if not url or not urlparse(url).netloc:
        return saves_done

    status = check_url(url)
    entry['url_status'] = status
    entry['url_checked_at'] = str(datetime.now().date())
    stats[status] += 1
    time.sleep(SLEEP_BETWEEN_CHECKS)

    if status == 'dead':
        logging.info(f"DEAD: {url}")
        if not entry.get('archived_url'):
            snap = wayback_lookup(url)
            if snap:
                entry['archived_url'] = snap
                stats['recovered'] += 1
                logging.info(f"  -> recovered from Wayback: {snap}")
            else:
                stats['unrecoverable'] += 1
                logging.warning(f"  -> NO archive available for {url}")
    elif ARCHIVE_LIVE and not entry.get('archived_url') and saves_done < MAX_SAVES_PER_RUN:
        snap = wayback_lookup(url)
        if snap:
            entry['archived_url'] = snap
        else:
            if wayback_save(url):
                stats['saved_now'] += 1
                saves_done += 1
                logging.info(f"  archived live URL: {url}")
            time.sleep(SLEEP_BETWEEN_SAVES)
    return saves_done


def main():
    with open(PROMISES_PATH) as f:
        data = json.load(f)

    stats = {'ok': 0, 'dead': 0, 'recovered': 0, 'unrecoverable': 0, 'saved_now': 0}
    saves_done = 0

    for p in data['promises']:
        # source link
        saves_done = process_entry(p, 'source_url', stats, saves_done)
        # homepage-only flag (a reader can't verify a promise from a homepage)
        path = urlparse(p.get('source_url', '')).path.strip('/')
        if not path:
            p['source_quality'] = 'homepage_only'
        # evidence links
        for a in p.get('evidence_articles', []):
            saves_done = process_entry(a, 'url', stats, saves_done)

    data['metadata']['links_last_checked'] = str(datetime.now().date())
    data['metadata']['link_check_stats'] = stats

    with open(PROMISES_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info(f"Done: {stats}")
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
