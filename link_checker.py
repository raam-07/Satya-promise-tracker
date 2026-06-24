import os
import json
import time
import logging
import requests
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Reuse the same archiving and search utilities from promise_tracker_pipeline.py
try:
    from promise_tracker_pipeline import (
        archive_url_flow,
        build_search_fallback_url
    )
except ImportError:
    # Fallback if imported from another path
    def archive_url_flow(url): return None, "none"
    def build_search_fallback_url(url, text): return f"https://www.google.com/search?q={url}"

PROMISES_JSON_PATH = os.environ.get('PROMISES_JSON_PATH', './promises.json')

def check_url_live(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    # Archive links are treated as always ok
    if "web.archive.org" in url or "archive.today" in url or "archive.ph" in url or "archive.is" in url:
        return True
        
    try:
        # Use HEAD first for speed
        res = requests.head(url, headers=headers, allow_redirects=True, timeout=10)
        if res.status_code in [200, 301, 302, 307, 308, 403, 405]:
            return True
        # If blocked or failed, fall back to GET (some sites block HEAD)
        res_get = requests.get(url, headers=headers, stream=True, timeout=10)
        if res_get.status_code in [200, 301, 302, 307, 308, 403, 405]:
            return True
    except Exception:
        pass
        
    return False

def run_checker():
    parser = argparse.ArgumentParser(description="Fast Satya Promise Link Checker")
    parser.add_argument('--skip-archive', action='store_true', help="Skip slow Wayback Machine/archive.today archiving")
    parser.add_argument('--concurrency', type=int, default=15, help="Number of concurrent check threads (default 15)")
    args = parser.parse_args()

    logging.info(f"Loading promises from {PROMISES_JSON_PATH}")
    if not os.path.exists(PROMISES_JSON_PATH):
        logging.critical(f"Promises file does not exist at {PROMISES_JSON_PATH}")
        sys.exit(1)
        
    with open(PROMISES_JSON_PATH, 'r', encoding='utf-8') as f:
        promises_data = json.load(f)
        
    promises = promises_data.get("promises", [])
    logging.info(f"Loaded {len(promises)} promises to process.")
    
    # Store initial IDs to enforce deletion guard
    initial_ids = {p["id"] for p in promises}
    
    # Collect all unique URLs that need checking
    urls_to_check = set()
    for p in promises:
        if "url" not in p:
            p["url"] = p.get("source_url", "")
        
        url = p.get("url", "")
        if url:
            urls_to_check.add(url)
            
        for e in p.get("evidence_articles", []):
            e_url = e.get("url", "")
            if e_url:
                urls_to_check.add(e_url)
                
    logging.info(f"Collected {len(urls_to_check)} unique URLs to check concurrently.")
    
    # Check URLs in parallel
    url_status_map = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        url_list = list(urls_to_check)
        logging.info(f"Spawning {args.concurrency} worker threads to verify links...")
        results = executor.map(check_url_live, url_list)
        for url, is_live in zip(url_list, results):
            url_status_map[url] = is_live
            
    logging.info("Link verification complete. Processing results...")
    
    stats_ok = 0
    stats_dead = 0
    stats_saved_now = 0
    stats_unrecoverable = 0
    
    # Cap: archive at most this many URLs per run to avoid IA rate limits.
    # Re-run the workflow multiple times to gradually archive everything.
    MAX_ARCHIVES_PER_RUN = 10
    archives_this_run = 0
    
    # We will throttle archiving attempts to avoid getting rate-limited
    throttle_delay = 5  # seconds
    
    for p_idx, p in enumerate(promises):
        url = p.get("url", "")
        
        # Check original live URL status using our pre-calculated map
        if url:
            is_live = url_status_map.get(url, False)
            p["url_status"] = "ok" if is_live else "dead"
        else:
            p["url_status"] = "dead"
            
        if p["url_status"] == "ok":
            stats_ok += 1
        else:
            stats_dead += 1
            if not p.get("archived_url"):
                stats_unrecoverable += 1
                
        # Handle search fallback URL
        if not p.get("search_fallback_url"):
            p["search_fallback_url"] = build_search_fallback_url(url, p.get("source_description", p.get("promise", "")))
            
        # Trigger archiving if needed and enabled
        if not args.skip_archive and p["url_status"] == "ok" and not p.get("archived_url"):
            if archives_this_run >= MAX_ARCHIVES_PER_RUN:
                logging.info(f"[{p_idx+1}/{len(promises)}] Archive limit ({MAX_ARCHIVES_PER_RUN}) reached for this run. Skipping: {url}")
            else:
                logging.info(f"[{p_idx+1}/{len(promises)}] Promise URL lacks archive. Archiving: {url}")
                time.sleep(throttle_delay)
                archived_url, archive_source = archive_url_flow(url)
                p["archived_url"] = archived_url
                p["archive_source"] = archive_source
                if archived_url:
                    stats_saved_now += 1
                    archives_this_run += 1
                    logging.info(f"Successfully archived: {archived_url} via {archive_source}")
                
        # Check evidence articles
        evidence_articles = p.get("evidence_articles", [])
        for e in evidence_articles:
            e_url = e.get("url", "")
            
            if e_url:
                e_live = url_status_map.get(e_url, False)
                e["url_status"] = "ok" if e_live else "dead"
            else:
                e["url_status"] = "dead"
                
            if e["url_status"] == "ok":
                stats_ok += 1
            else:
                stats_dead += 1
                if not e.get("archived_url"):
                    stats_unrecoverable += 1
                    
            if "supporting_quote" not in e:
                e["supporting_quote"] = e.get("quote", "")
                
            if not e.get("search_fallback_url"):
                e["search_fallback_url"] = build_search_fallback_url(e_url, e.get("title", ""))
                
            # Trigger archiving if needed and enabled
            if not args.skip_archive and e["url_status"] == "ok" and not e.get("archived_url"):
                if archives_this_run >= MAX_ARCHIVES_PER_RUN:
                    logging.info(f"  Archive limit ({MAX_ARCHIVES_PER_RUN}) reached for this run. Skipping: {e_url}")
                else:
                    logging.info(f"  Evidence URL lacks archive. Archiving: {e_url}")
                    time.sleep(throttle_delay)
                    archived_url, archive_source = archive_url_flow(e_url)
                    e["archived_url"] = archived_url
                    e["archive_source"] = archive_source
                    if archived_url:
                        stats_saved_now += 1
                        archives_this_run += 1
                        logging.info(f"  Successfully archived: {archived_url} via {archive_source}")
                    
        p["url_checked_at"] = time.strftime("%Y-%m-%d")
        
    # Enforce deletion guard before saving
    final_ids = {p["id"] for p in promises}
    removed = initial_ids - final_ids
    if removed:
        logging.critical(f"GUARD BLOCK: Link checker would delete {len(removed)} promises: {removed}")
        sys.exit(1)
        
    # Update metadata
    if "metadata" not in promises_data:
        promises_data["metadata"] = {}
        
    promises_data["metadata"]["last_updated"] = time.strftime("%Y-%m-%d")
    promises_data["metadata"]["links_last_checked"] = time.strftime("%Y-%m-%d")
    promises_data["metadata"]["total_promises"] = len(promises)
    promises_data["metadata"]["promises_with_evidence"] = sum(1 for p in promises if len(p.get("evidence_articles", [])) > 0)
    
    old_stats = promises_data["metadata"].get("link_check_stats", {})
    old_dead = old_stats.get("dead", 0)
    recovered = 0
    if old_dead > stats_dead:
        recovered = old_dead - stats_dead
        
    promises_data["metadata"]["link_check_stats"] = {
        "ok": stats_ok,
        "dead": stats_dead,
        "recovered": old_stats.get("recovered", 0) + recovered,
        "unrecoverable": stats_unrecoverable,
        "saved_now": stats_saved_now
    }
    
    # Save promises.json in-place
    logging.info(f"Saving updated promises data back to {PROMISES_JSON_PATH}")
    with open(PROMISES_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(promises_data, f, indent=2, ensure_ascii=False)
        
    logging.info("Link checking completed successfully!")

if __name__ == "__main__":
    run_checker()
