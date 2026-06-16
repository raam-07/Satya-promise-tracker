import os
import json
import time
import logging
import re
import requests
import sqlite3
import zlib
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuration & Defaults
MODEL_2B_PATH = os.environ.get('MODEL_2B_PATH', './models/gemma-2-2b-it-Q4_K_M.gguf')
MODEL_9B_PATH = os.environ.get('MODEL_9B_PATH', './models/gemma-2-9b-it-Q4_K_M.gguf')
PROMISES_JSON_PATH = os.environ.get('PROMISES_JSON_PATH', './promises.json')

default_db_path = '/Users/mac/Downloads/Code/Satya/satya.db'
if not os.path.exists(os.path.dirname(default_db_path)):
    default_db_path = os.path.join(os.path.dirname(__file__), 'satya.db')
DB_PATH = os.environ.get('SATYA_DB_PATH', default_db_path)

# Wayback Machine Archiver Helper
def archive_url_wayback(url):
    """
    Triggers an automated snapshot on the Wayback Machine.
    Returns the archived URL if successful, otherwise the original URL.
    """
    logging.info(f"Triggering Wayback Machine archive for: {url}")
    save_url = f"https://web.archive.org/save/{url}"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.post(save_url, headers=headers, timeout=15)
        if response.status_code == 200:
            # Check headers or construct fallback archive URL
            # The Wayback Save API redirects or places archive link in the headers
            location = response.headers.get('Content-Location') or response.headers.get('Location')
            if location:
                archive_link = f"https://web.archive.org{location}" if location.startswith('/') else location
                logging.info(f"Successfully archived! Link: {archive_link}")
                return archive_link
            
            # Fallback format: https://web.archive.org/web/YYYYMMDDhhmmss/URL
            timestamp = time.strftime("%Y%m%d%H%M%S")
            archive_link = f"https://web.archive.org/web/{timestamp}/{url}"
            logging.info(f"Archive triggered (fallback link): {archive_link}")
            return archive_link
        else:
            logging.warning(f"Wayback responded with status {response.status_code}. Using original URL.")
    except Exception as e:
        logging.error(f"Failed to save URL to Wayback: {e}")
    return url

# DB connection helper
def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            logging.info(f"Connecting to remote Turso Database at: {db_url}")
            return libsql.connect(database=db_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local SQLite.")
            
    logging.info(f"Connecting to local SQLite Database at: {DB_PATH}")
    return sqlite3.connect(DB_PATH)

# Promises JSON handling helpers
def load_promises():
    if os.path.exists(PROMISES_JSON_PATH):
        try:
            with open(PROMISES_JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to parse promises.json: {e}")
    
    # Return a default empty schema structure if not found
    return {
        "metadata": {
            "version": "1.0",
            "last_updated": time.strftime("%Y-%m-%d"),
            "description": "Satya Promise Tracker — key promises made by Indian political leaders",
            "maintainer": "Satya Project",
            "total_promises": 0,
            "promises_with_evidence": 0,
            "last_processed_row": 0
        },
        "promises": []
    }

def save_promises(data):
    try:
        with open(PROMISES_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved data to: {PROMISES_JSON_PATH}")
    except Exception as e:
        logging.error(f"Failed to write to promises.json: {e}")

# Regex Screening Helper
def regex_pre_screen(title, content):
    text = (title + " " + content).lower()
    
    # Scan for years ranging from 2014 to 2035
    has_year = any(str(yr) in text for yr in range(2014, 2036))
    if not has_year:
        return False
        
    # Standard keywords describing future goals, timelines, updates, or verdicts
    keywords = ["promise", "pledge", "target", "launch", "guarantee", "subsidy", 
                "welfare", "scheme", "deadline", "verdict", "manifesto", "sankalp"]
    return any(kw in text for kw in keywords)

# LLM Loading Helper
def init_llm(model_path, ctx_size):
    try:
        from llama_cpp import Llama
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}")
            return None
        
        logging.info(f"Loading local GGUF model: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_batch=512,
            n_threads=4,  # Optimized for 4 vCPU runner
            verbose=False
        )
        return llm
    except Exception as e:
        logging.error(f"Failed to load GGUF model from {model_path}: {e}")
        return None

# ==============================================================================
# --- THE 3-STAGE PIPELINE ---
# ==============================================================================

def run_stage1_noise_filter(llm_2b, title, content):
    """
    STAGE 1: Noise Filter (Gemma 2B)
    Binary classification to verify if the article contains a real promise or progress update.
    """
    prompt = f"""<start_of_turn>user
Determine if the following news article contains specific information about a political promise made by an Indian politician, or an active progress/status update about an existing promise.

CRITICAL RULES:
- Reply ONLY with "YES" if there is a concrete promise or progress update mentioned.
- Reply ONLY with "NO" if the article is general news, opinion, debate, or does not contain a specific promise/timeline update.
- Do not write any explanation, introduction, or other characters.

Title: {title}
Article snippet: {content[:1500]}
<end_of_turn>
<start_of_turn>model
"""
    try:
        output = llm_2b(prompt, max_tokens=10, temperature=0.0)
        res = output['choices'][0]['text'].strip().upper()
        logging.info(f"Stage 1 Noise Filter result: {res}")
        return "YES" in res
    except Exception as e:
        logging.error(f"Stage 1 Noise Filter error: {e}")
        return False

def run_stage2_extractor(llm_9b, title, content, existing_promises):
    """
    STAGE 2: Structured Extractor (Gemma 9B)
    Extracts structured JSON payload representing the promise, target, and status.
    """
    promises_context = ""
    for p in existing_promises:
        promises_context += f"- ID: {p['id']}, Politician: {p['person']}, Promise: \"{p['promise']}\", Current Status: {p['status']}\n"

    prompt = f"""<start_of_turn>user
Analyze the news article below and extract political promise information. 
Output ONLY a raw JSON object matching the schema. Do not wrap in markdown code blocks.

JSON SCHEMA:
{{
  "politician": "Name of the politician making the promise",
  "party": "Political party (e.g. BJP, Congress, AAP, TMC, etc.)",
  "promise_text": "Exact promise title or goal (keep it concise, e.g. 'Build 20,000 houses')",
  "deadline_year": "YYYY target year, or 'ongoing'",
  "is_new_promise": true if this is a brand new promise, false if it updates one of the existing promises below,
  "matched_existing_promise_id": "pXXX ID if it matches an existing promise, otherwise null",
  "verdict": "kept, broken, or ongoing based on the article facts",
  "reasoning": "1-2 sentence explanation of the status/verdict based on the article text"
}}

Existing Promises to Match Against:
{promises_context}

Article Title: {title}
Article Content: {content[:2500]}
<end_of_turn>
<start_of_turn>model
"""
    try:
        output = llm_9b(prompt, max_tokens=350, temperature=0.0)
        json_text = output['choices'][0]['text'].strip()
        
        # Clean any accidental markdown code wrappers
        if json_text.startswith("```"):
            lines = json_text.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                json_text = "\n".join(lines[1:-1]).strip()
                
        return json.loads(json_text)
    except Exception as e:
        logging.error(f"Stage 2 Extractor error/invalid JSON: {e}")
        return None

def run_stage3_critic(llm_9b, original_content, proposed_json):
    """
    STAGE 3: Adversarial Critic (Gemma 9B with different prompt)
    Audits the extraction JSON against original text to reject hallucinations or vague items.
    """
    prompt = f"""<start_of_turn>user
Adversarial Audit Task: Review the original article and the proposed extraction JSON.
Ensure absolute logical alignment and facts.

Original Article: {original_content[:2000]}
Proposed JSON: {json.dumps(proposed_json, indent=2)}

Audit Checks:
1. Is the politician's name correct and explicitly stated in the article?
2. Is the extracted target deadline year explicitly mentioned or logically clear in the article?
3. Is the verdict (kept/broken/ongoing) mathematically/logically aligned with the article facts? (e.g., target passed with no result = broken, target successfully finished = kept).
4. If the promise is vague, rhetorical, or an opinion, you MUST reject it.

If the extraction is correct and supported, reply ONLY with "APPROVED".
If there is any error, hallucination, or vague target, reply ONLY with "REJECTED: [Brief reason]".
<end_of_turn>
<start_of_turn>model
"""
    try:
        output = llm_9b(prompt, max_tokens=50, temperature=0.0)
        res = output['choices'][0]['text'].strip()
        logging.info(f"Stage 3 Critic result: {res}")
        return res.startswith("APPROVED"), res
    except Exception as e:
        logging.error(f"Stage 3 Critic error: {e}")
        return False, f"REJECTED: critic logic failure: {e}"

# ==============================================================================
# --- MAIN PIPELINE EXECUTION ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Satya Automated Local Promise Tracker Pipeline")
    parser.add_argument('--batch-size', type=int, default=500, help="Number of articles to process in this run (default 500)")
    parser.add_argument('--reset-pointer', action='store_true', help="Reset the last_processed_row pointer to 0 and scan from beginning")
    parser.add_argument('--dry-run', action='store_true', help="Run without writing changes to promises.json or archiving URLs")
    args = parser.parse_args()

    logging.info("Starting Satya Promise Tracker Pipeline...")
    
    # 1. Load existing promises data
    promises_data = load_promises()
    
    last_processed = promises_data["metadata"].get("last_processed_row", 0)
    if args.reset_pointer:
        last_processed = 0
        logging.info("Pointer reset requested. Starting from row ID 0.")
    else:
        logging.info(f"Resuming from last processed row ID: {last_processed}")

    # 2. Connect to Database (reads from Turso remote if variables are set)
    try:
        conn = get_db_connection()
    except Exception as e:
        logging.critical(f"Failed to connect to database: {e}")
        sys.exit(1)

    # 3. Load Gemma Models once (lazy loading when first required, then persistent)
    llm_2b = None
    llm_9b = None

    highest_processed_id = last_processed
    new_promise_count = 0
    updated_promise_count = 0

    # Fetch batch of classified articles
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, url, scraped_at, content, rephrased_article, ministers_mentioned, party_mentioned, states_mentioned
        FROM articles
        WHERE status IN ('classified', 'entity_processed')
        ORDER BY id ASC
        LIMIT ?
    """, (args.batch_size,))
    rows = cursor.fetchall()
    
    if not rows:
        logging.info("No new classified articles found to process.")
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write("has_more=false\n")
        conn.close()
        sys.exit(0)
        
    logging.info(f"Retrieved {len(rows)} classified articles to evaluate.")

    # We only load the LLM models if not loaded and at least one article passes the regex pre-screen
    pre_screened_rows = []
    for r in rows:
        article_id, title, url, scraped_at, compressed_content, compressed_rephrased, ministers_str, party_str, states_str = r
        
        try:
            content = zlib.decompress(compressed_content).decode('utf-8') if compressed_content else ""
        except Exception:
            content = ""
            
        if regex_pre_screen(title, content):
            pre_screened_rows.append((r, content))
            
    logging.info(f"Regex screening complete: {len(pre_screened_rows)} out of {len(rows)} articles passed to Stage 1.")

    if pre_screened_rows:
        llm_2b = init_llm(MODEL_2B_PATH, 2048)
        llm_9b = init_llm(MODEL_9B_PATH, 4096)
        
        if not llm_2b or not llm_9b:
            logging.critical("Failed to load local Gemma engines. Exiting pipeline.")
            conn.close()
            sys.exit(1)

    # Process each pre-screened article in this batch
    for r_data, content in pre_screened_rows:
        r = r_data
        article_id, title, url, scraped_at, _, compressed_rephrased, _, _, _ = r
        
        try:
            rephrased = zlib.decompress(compressed_rephrased).decode('utf-8') if compressed_rephrased else content
        except Exception:
            rephrased = content

        logging.info(f"\n--- Evaluating Article ID {article_id}: {title[:60]}... ---")

        # STAGE 1: Noise Filter
        if not run_stage1_noise_filter(llm_2b, title, rephrased):
            logging.info("Stage 1 Noise Filter: Discarded as noise.")
            continue

        # STAGE 2: Structured Extractor
        extracted_json = run_stage2_extractor(llm_9b, title, rephrased, promises_data["promises"])
        if not extracted_json:
            logging.info("Stage 2 Extractor: Failed to parse valid JSON payload.")
            continue

        logging.info(f"Stage 2 Extractor Draft JSON:\n{json.dumps(extracted_json, indent=2)}")

        # STAGE 3: Critic Check
        approved, critic_msg = run_stage3_critic(llm_9b, rephrased, extracted_json)
        if not approved:
            logging.info(f"Stage 3 Critic Rejected: {critic_msg}")
            continue

        logging.info("Stage 3 Critic Approved! Committing changes...")

        # Wayback Archiving (skip if dry-run)
        final_url = url
        if not args.dry_run:
            final_url = archive_url_wayback(url)

        # Map to evidence article format
        evidence_item = {
            "url": final_url,
            "title": title,
            "source": extracted_json.get("source", "News Article"),
            "scraped_at": time.strftime("%Y-%m-%d", time.localtime(scraped_at)) if scraped_at else time.strftime("%Y-%m-%d"),
            "relevance_score": 100,
            "gemma_validated": True,
            "rephrased": rephrased[:300] + "...",
            "content": content[:400] + "..."
        }

        # Update JSON schema structures
        is_new = extracted_json.get("is_new_promise", True)
        matched_id = extracted_json.get("matched_existing_promise_id")

        if not is_new and matched_id:
            # Update existing promise
            found = False
            for p in promises_data["promises"]:
                if p["id"] == matched_id:
                    # Append evidence
                    if "evidence_articles" not in p:
                        p["evidence_articles"] = []
                    
                    # Prevent duplicate URLs
                    if not any(e["url"] == final_url for e in p["evidence_articles"]):
                        p["evidence_articles"].append(evidence_item)
                        
                    p["status"] = extracted_json.get("verdict", p["status"])
                    p["status_last_reviewed"] = time.strftime("%Y-%m-%d")
                    p["gemma_suggestion"] = extracted_json.get("verdict", p.get("gemma_suggestion"))
                    p["gemma_reasoning"] = extracted_json.get("reasoning", p.get("gemma_reasoning"))
                    p["evidence_count"] = len(p["evidence_articles"])
                    found = True
                    updated_promise_count += 1
                    logging.info(f"Updated existing promise ID: {matched_id}")
                    break
            if not found:
                logging.warning(f"Extracted matched ID {matched_id} not found in promises.json. Treating as new.")
                is_new = True

        if is_new:
            # Generate next sequential ID
            existing_ids = [int(p["id"][1:]) for p in promises_data["promises"] if p["id"].startswith('p')]
            next_id_num = max(existing_ids) + 1 if existing_ids else 1
            next_id = f"p{next_id_num:03d}"

            new_promise = {
                "id": next_id,
                "person": extracted_json.get("politician", "Unknown Politician"),
                "party": extracted_json.get("party", "Unknown Party"),
                "role": "Politician",
                "promise": extracted_json.get("promise_text", title),
                "category": "general",
                "made_on": time.strftime("%Y-%m-%d"),
                "deadline": extracted_json.get("deadline_year", "ongoing"),
                "source_url": final_url,
                "source_description": title,
                "status": extracted_json.get("verdict", "ongoing"),
                "status_last_reviewed": time.strftime("%Y-%m-%d"),
                "gemma_suggestion": extracted_json.get("verdict", "ongoing"),
                "gemma_reasoning": extracted_json.get("reasoning", ""),
                "evidence_articles": [evidence_item],
                "notes": extracted_json.get("reasoning", ""),
                "gemma_confidence": "high",
                "gemma_assessed_at": time.strftime("%Y-%m-%d"),
                "created_at": time.strftime("%Y-%m-%d"),
                "url_status": "ok",
                "url_checked_at": time.strftime("%Y-%m-%d"),
                "promise_type": "specific",
                "evidence_count": 1
            }
            promises_data["promises"].append(new_promise)
            new_promise_count += 1
            logging.info(f"Created new promise ID: {next_id}")

    # Track pointer even if no article passed filters, so we don't scan them again next time
    for r in rows:
        highest_processed_id = max(highest_processed_id, r[0])

    # Track pointer and save metadata pointer after every batch to prevent progress loss
    promises_data["metadata"]["last_processed_row"] = highest_processed_id
    promises_data["metadata"]["last_updated"] = time.strftime("%Y-%m-%d")
    promises_data["metadata"]["total_promises"] = len(promises_data["promises"])
    promises_data["metadata"]["promises_with_evidence"] = sum(1 for p in promises_data["promises"] if p.get("evidence_count", 0) > 0)

    # Mark all articles fetched in this batch as 'processed' in the database
    if not args.dry_run and rows:
        try:
            db_cursor = conn.cursor()
            article_ids = [r[0] for r in rows]
            placeholders = ",".join("?" for _ in article_ids)
            db_cursor.execute(f"UPDATE articles SET status = 'processed' WHERE id IN ({placeholders})", article_ids)
            conn.commit()
            logging.info(f"Database updated: Marked {len(rows)} articles as processed.")
        except Exception as e:
            logging.error(f"Failed to update article status in database: {e}")

    if not args.dry_run:
        save_promises(promises_data)
        logging.info(f"Progress saved: pointer advanced to row ID {highest_processed_id}")

    # Write has_more output for self-trigger loop in GitHub Actions
    has_more = "true" if len(rows) == args.batch_size else "false"
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"has_more={has_more}\n")
    logging.info(f"Set workflow output has_more={has_more}")

    if not args.dry_run:
        logging.info(f"Pipeline complete: {new_promise_count} new promises added, {updated_promise_count} promises updated.")
    else:
        logging.info(f"Dry run complete. No modifications saved. (Discovered {new_promise_count} new, {updated_promise_count} updates).")

    conn.close()

if __name__ == "__main__":
    main()
