import os
import json
import re
import sys

# Add parent directory to sys.path so we can import from pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promise_tracker_pipeline import build_search_fallback_url

PROMISES_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'promises.json')

def migrate():
    print(f"Loading {PROMISES_JSON_PATH}")
    with open(PROMISES_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    promises = data.get("promises", [])
    print(f"Loaded {len(promises)} promises.")
    
    for idx, p in enumerate(promises):
        source_url = p.get("source_url", "")
        
        # Extract archive link if source_url is a Wayback URL
        archived_url = p.get("archived_url", "")
        archive_source = p.get("archive_source", "")
        
        original_live_url = source_url
        if "web.archive.org/web/" in source_url:
            match = re.search(r'/web/\d+/(https?://.+)$', source_url)
            if match:
                original_live_url = match.group(1)
                if not archived_url:
                    archived_url = source_url
                    archive_source = "wayback"
                    
        if not archive_source:
            if archived_url:
                if "web.archive.org" in archived_url:
                    archive_source = "wayback"
                elif any(d in archived_url for d in ["archive.ph", "archive.today", "archive.is"]):
                    archive_source = "archivetoday"
                else:
                    archive_source = "none"
            else:
                archived_url = ""
                archive_source = "none"
                
        p["url"] = original_live_url
        p["url_status"] = p.get("url_status", "ok")
        p["archived_url"] = archived_url
        p["archive_source"] = archive_source
        
        if not p.get("search_fallback_url"):
            p["search_fallback_url"] = build_search_fallback_url(original_live_url, p.get("source_description", p.get("promise", "")))
            
        if "supporting_quote" not in p:
            p["supporting_quote"] = p.get("quote", "")
            
        # Evidence articles
        for e in p.get("evidence_articles", []):
            e_url = e.get("url", "")
            e_archived = e.get("archived_url", "")
            e_source = e.get("archive_source", "")
            
            e_live_url = e_url
            if "web.archive.org/web/" in e_url:
                e_match = re.search(r'/web/\d+/(https?://.+)$', e_url)
                if e_match:
                    e_live_url = e_match.group(1)
                    if not e_archived:
                        e_archived = e_url
                        e_source = "wayback"
                        
            if not e_source:
                if e_archived:
                    if "web.archive.org" in e_archived:
                        e_source = "wayback"
                    elif any(d in e_archived for d in ["archive.ph", "archive.today", "archive.is"]):
                        e_source = "archivetoday"
                    else:
                        e_source = "none"
                else:
                    e_archived = ""
                    e_source = "none"
                    
            e["url"] = e_live_url
            e["url_status"] = e.get("url_status", "ok")
            e["archived_url"] = e_archived
            e["archive_source"] = e_source
            
            if not e.get("search_fallback_url"):
                e["search_fallback_url"] = build_search_fallback_url(e_live_url, e.get("title", ""))
                
            if "supporting_quote" not in e:
                e["supporting_quote"] = e.get("quote", "")
                
    print("Saving migrated promises...")
    with open(PROMISES_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Migration finished!")

if __name__ == "__main__":
    migrate()
