import os
import json
import time
import logging
import requests
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Reuse the same archiving and search utilities from promise_tracker_pipeline.py
from promise_tracker_pipeline import (
    archive_url_flow,
    build_search_fallback_url
)

PROMISES_JSON_PATH = os.environ.get('PROMISES_JSON_PATH', './promises.json')

def check_url_live(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    # Archive links are treated as always ok
    if "web.archive.org" in url or "archive.today" in url or "archive.ph" in url or "archive.is" in url:
        return True
        
    try:
        logging.info(f"Checking URL status (HEAD): {url}")
        res = requests.head(url, headers=headers, allow_redirects=True, timeout=10)
        if res.status_code in [200, 301, 302, 307, 308]:
            return True
        # If it returns 403, 405, etc. (some sites block HEAD requests), fall back to GET
        logging.info(f"HEAD failed with status {res.status_code}. Retrying with GET: {url}")
        res_get = requests.get(url, headers=headers, stream=True, timeout=10)
        if res_get.status_code in [200, 301, 302, 307, 308]:
            return True
    except Exception as e:
        logging.warning(f"Error checking live status of {url}: {e}")
        
    return False

def run_checker():
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
    
    stats_ok = 0
    stats_dead = 0
    stats_saved_now = 0
    stats_unrecoverable = 0
    
    # We will throttle archiving attempts to avoid getting rate-limited
    throttle_delay = 5  # seconds
    
    for p_idx, p in enumerate(promises):
        logging.info(f"[{p_idx+1}/{len(promises)}] Processing promise ID: {p['id']} ({p.get('person')})")
        
        # 1. Ensure primary fields exist on the promise
        # Maintain backward compatibility with source_url
        if "url" not in p:
            p["url"] = p.get("source_url", "")
            
        url = p.get("url", "")
        
        # Determine initial url_status if not present
        if "url_status" not in p:
            p["url_status"] = "ok"
            
        # Check original live URL status
        is_live = False
        if url:
            is_live = check_url_live(url)
            p["url_status"] = "ok" if is_live else "dead"
            
        if p["url_status"] == "ok":
            stats_ok += 1
        else:
            stats_dead += 1
            if not p.get("archived_url"):
                stats_unrecoverable += 1
                
        # Handle search fallback URL
        if not p.get("search_fallback_url"):
            p["search_fallback_url"] = build_search_fallback_url(url, p.get("source_description", p.get("promise", "")))
            logging.info(f"Generated search fallback URL for promise: {p['search_fallback_url']}")
            
        # If url is live but archived_url is missing, trigger archiving
        if p["url_status"] == "ok" and not p.get("archived_url"):
            logging.info(f"Promise URL lacks archive. Triggering flow...")
            time.sleep(throttle_delay)
            archived_url, archive_source = archive_url_flow(url)
            p["archived_url"] = archived_url
            p["archive_source"] = archive_source
            if archived_url:
                stats_saved_now += 1
                logging.info(f"Successfully archived: {archived_url} via {archive_source}")
            else:
                logging.info("Archiving returned empty results.")
                
        # 2. Check evidence articles for the promise
        evidence_articles = p.get("evidence_articles", [])
        for e_idx, e in enumerate(evidence_articles):
            logging.info(f"  Checking evidence article [{e_idx+1}/{len(evidence_articles)}]: {e.get('title')}")
            
            e_url = e.get("url", "")
            if not e.get("url_status"):
                e["url_status"] = "ok"
                
            e_live = False
            if e_url:
                e_live = check_url_live(e_url)
                e["url_status"] = "ok" if e_live else "dead"
                
            if e["url_status"] == "ok":
                stats_ok += 1
            else:
                stats_dead += 1
                if not e.get("archived_url"):
                    stats_unrecoverable += 1
                    
            # Ensure supporting_quote copy
            if "supporting_quote" not in e:
                e["supporting_quote"] = e.get("quote", "")
                
            # Ensure search fallback URL exists
            if not e.get("search_fallback_url"):
                e["search_fallback_url"] = build_search_fallback_url(e_url, e.get("title", ""))
                logging.info(f"  Generated search fallback URL for evidence: {e['search_fallback_url']}")
                
            # Trigger archive if live and missing archive url
            if e["url_status"] == "ok" and not e.get("archived_url"):
                logging.info(f"  Evidence URL lacks archive. Triggering flow...")
                time.sleep(throttle_delay)
                archived_url, archive_source = archive_url_flow(e_url)
                e["archived_url"] = archived_url
                e["archive_source"] = archive_source
                if archived_url:
                    stats_saved_now += 1
                    logging.info(f"  Successfully archived: {archived_url} via {archive_source}")
                else:
                    logging.info("  Archiving returned empty results.")
                    
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
    
    # Calculate recovery stats if possible
    # We can fetch old stats or estimate
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
