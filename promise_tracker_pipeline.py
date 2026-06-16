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

# Politician Name Validation Helper
def is_valid_indian_politician(name):
    if not name:
        return False
    
    name_lower = name.lower().strip()
    
    # Block list of generic administrative/institutional titles or non-person entities
    generic_blocklist = [
        "government", "ministry", "cabinet", "court", "judge", "assembly", "parliament",
        "committee", "board", "commission", "department", "administration", "authority",
        "university", "vice-chancellor", "vice chancellor", "principal", "director",
        "police", "commissioner", "officer", "secretary", "spokesperson", "advisor",
        "governor", "bank", "rbi", "center", "state", "hc", "sc", "supreme court",
        "high court", "panchayat", "corporation", "municipality", "bureaucracy"
    ]
    
    for term in generic_blocklist:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, name_lower):
            logging.info(f"Rejected generic/administrative politician name: {name}")
            return False
        
    # Check length and ensure it contains alphabetic characters
    if len(name_lower) < 3 or len(name_lower) > 50:
        logging.info(f"Rejected politician name due to length: {name}")
        return False
        
    if not re.search(r'[a-zA-Z]', name):
        return False
        
    return True

# Load Known Politician Registry
def load_known_politicians():
    """
    Loads known politician names and aliases from entities.json (via GitHub raw URL or local file).
    """
    names = set()
    url = "https://raw.githubusercontent.com/raam-07/satya-entity-library/main/entities.json"
    local_path = "../satya-entity-library/entities.json"
    
    data = None
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info("Loaded entities.json from local path.")
        except Exception as e:
            logging.warning(f"Failed to load local entities.json: {e}")
            
    if not data:
        try:
            logging.info(f"Fetching entities.json from remote: {url}")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
        except Exception as e:
            logging.warning(f"Failed to fetch remote entities.json: {e}")
            
    if data and "india" in data:
        for cat in ['cabinet_ministers', 'opposition_leaders', 'state_chief_ministers', 'generic_politicians']:
            for p in data['india'].get(cat, []):
                names.add(p['name'].lower().strip())
                for alias in p.get('aliases', []):
                    names.add(alias.lower().strip())
                    
    # Seed with core/newly validated politicians just in case network/file fetch fails
    core_names = [
        "narendra modi", "amit shah", "arvind kejriwal", "rahul gandhi", "yogi adityanath",
        "mamata banerjee", "siddaramaiah", "nitin gadkari", "d.k. shivakumar", "d.k. suresh",
        "bhajan lal sharma", "revanth reddy", "stalin", "m.k. stalin", "chandrababu naidu"
    ]
    for n in core_names:
        names.add(n)
        
    return names

def is_known_politician(name, known_set):
    if not name:
        return False
    name_clean = name.lower().strip()
    
    # Direct match
    if name_clean in known_set:
        return True
        
    # Substring match (e.g. "PM Narendra Modi" matches "narendra modi")
    for k in known_set:
        if k in name_clean or name_clean in k:
            return True
            
    return False



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
You are a strict filter for an INDIAN political accountability tracker.
Reply "YES" only if BOTH are true:
1. The article states a concrete promise, pledge, or a progress/status update on one.
2. It was made by a specific, NAMED INDIAN politician (PM, CM, minister, MP, MLA, party leader).

Reply "NO" if:
- The speaker is a foreign politician or foreign official.
- The "speaker" is an institution, department, government, court, or office — not a named person.
- It is general news, opinion, debate, or has no concrete promise.

Reply with ONLY one word: YES or NO. No explanation.

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
Analyze the article and extract political promise information.
Output ONLY a raw JSON object matching the schema. No markdown.

CRITICAL RULES:
1. 'politician' MUST be a specific, named Indian political leader (e.g. "M.K. Stalin", "Siddaramaiah").
2. NEVER extract government bodies/departments ("Tamil Nadu government", "Cabinet", "Ministry of Finance").
3. NEVER extract non-political officials (Vice-Chancellor, bureaucrats, police, judges).
4. NEVER extract foreign/international politicians.
5. The promise MUST have been made by this politician PERSONALLY. If the article's promise was made by someone else quoted in it, return {{}}.
6. 'supporting_quote' MUST be copied word-for-word from the article — the exact sentence where the promise/update appears. If you cannot find such a sentence, return {{}}.
7. If no valid promise from a named Indian politician, return {{}}.

JSON SCHEMA:
{{
  "politician": "Name of the Indian politician",
  "party": "Their party (BJP, Congress, AAP, DMK, TDP, etc.) — use the well-known party if not stated",
  "promise_text": "Concise promise/goal (e.g. 'Build 20,000 houses')",
  "supporting_quote": "the EXACT sentence from the article, copied verbatim",
  "deadline_year": "YYYY or 'ongoing'",
  "is_new_promise": true or false,
  "matched_existing_promise_id": "pXXX or null",
  "verdict": "kept, broken, or ongoing",
  "reasoning": "1-2 sentences grounded only in the article"
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
                
        data = json.loads(json_text)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                return None
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        logging.error(f"Stage 2 Extractor error/invalid JSON: {e}")
        return None


def run_stage3_critic(llm_9b, original_content, proposed_json):
    """
    STAGE 3: Adversarial Critic (Gemma 9B with different prompt)
    Audits the extraction JSON against original text to reject hallucinations or vague items.
    """
    prompt = f"""<start_of_turn>user
Adversarial audit. Review the original article against the proposed JSON.

Original Article: {original_content[:2000]}
Proposed JSON: {json.dumps(proposed_json, indent=2)}

Reject (reply "REJECTED: [reason]") if ANY check fails:
1. 'politician' is not a specific named Indian politician — reject generic bodies ("Tamil Nadu government"), non-politicians ("Vice-Chancellor"), or foreign figures ("Yao Ming").
2. The politician's name is not explicitly stated in the article.
3. 'supporting_quote' does NOT appear word-for-word in the article, OR does not actually contain the promise. (This is the most important check — reject fabricated quotes.)
4. The promise was actually made by someone else in the article, not this politician.
5. The promise is vague, rhetorical, or opinion.
6. The verdict contradicts the article facts.

If everything passes, reply ONLY "APPROVED". Otherwise "REJECTED: [brief reason]".
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
    
    # 1. Load existing promises data and known politician entities registry
    promises_data = load_promises()
    known_politicians = load_known_politicians()
    
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
        SELECT a.id, a.title, a.url, a.scraped_at, a.content, a.rephrased_article, a.ministers_mentioned, a.party_mentioned, a.states_mentioned, s.name AS source_name
        FROM articles a
        LEFT JOIN sources s ON a.source_id = s.id
        WHERE a.status = 'entity_processed'
        ORDER BY a.id ASC
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

    # We only load the LLM models if not loaded and at least one article passes the filters
    pre_screened_rows = []
    for r in rows:
        article_id, title, url, scraped_at, compressed_content, compressed_rephrased, ministers_str, party_str, states_str, source_name = r
        
        try:
            content = zlib.decompress(compressed_content).decode('utf-8') if compressed_content else ""
        except Exception:
            content = ""
            
        # Check if the article mentions any known registered politician from entities.json
        has_known_politician = False
        ministers_list = []
        if ministers_str:
            try:
                ministers_list = json.loads(ministers_str)
            except Exception:
                ministers_list = []
                
        if isinstance(ministers_list, list) and ministers_list:
            for m in ministers_list:
                if is_known_politician(m, known_politicians):
                    has_known_politician = True
                    break
                    
        # Fallback: substring search of the known politicians/aliases in the title or content
        if not has_known_politician:
            text_to_scan = (title + " " + content).lower()
            for kp in known_politicians:
                if len(kp) >= 4 and kp in text_to_scan:
                    has_known_politician = True
                    break
                elif len(kp) < 4:
                    if re.search(r'\b' + re.escape(kp) + r'\b', text_to_scan):
                        has_known_politician = True
                        break
                        
        if not has_known_politician:
            logging.info(f"Article ID {article_id} skipped: No known registered Indian politician mentioned.")
            continue
            
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
        article_id, title, url, scraped_at, _, compressed_rephrased, _, _, _, source_name = r
        
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

        # 1. Hard check: Politician verification
        politician_name = extracted_json.get("politician", "")
        if not is_valid_indian_politician(politician_name) or not is_known_politician(politician_name, known_politicians):
            logging.warning(f"Stage 2 Extractor: Discarding due to invalid or unregistered politician '{politician_name}'.")
            continue

        # 2. Hard check: Supporting quote verbatim match
        supporting_quote = extracted_json.get("supporting_quote", "")
        if not supporting_quote:
            logging.warning("Stage 2 Extractor: Discarding due to missing supporting_quote.")
            continue
            
        norm_quote = " ".join(supporting_quote.lower().split())
        norm_rephrased = " ".join(rephrased.lower().split())
        norm_full_content = " ".join(content.lower().split())
        
        if norm_quote not in norm_rephrased and norm_quote not in norm_full_content:
            logging.warning(f"Stage 2 Extractor: Discarding because supporting_quote '{supporting_quote}' does not appear verbatim in article.")
            continue

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
            "source": source_name if source_name else "News Article",
            "scraped_at": time.strftime("%Y-%m-%d", time.localtime(scraped_at)) if scraped_at else time.strftime("%Y-%m-%d"),
            "relevance_score": 100,
            "gemma_validated": True,
            "rephrased": rephrased[:300] + "...",
            "content": content[:400] + "...",
            "quote": extracted_json.get("supporting_quote", "")
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
                    p["supporting_quote"] = extracted_json.get("supporting_quote", p.get("supporting_quote", ""))
                    p["evidence_count"] = len(p["evidence_articles"])
                    found = True
                    updated_promise_count += 1
                    logging.info(f"Updated existing promise ID: {matched_id}")
                    break
            if not found:
                logging.warning(f"Extracted matched ID {matched_id} not found in promises.json. Treating as new.")
                is_new = True

        if is_new:
            politician_name = extracted_json.get("politician", "Unknown Politician")
            if not is_valid_indian_politician(politician_name):
                logging.warning(f"Skipping promise extraction: Invalid/generic political entity name: '{politician_name}'")
                continue

            if not is_known_politician(politician_name, known_politicians):
                logging.warning(f"Skipping promise extraction: Politician '{politician_name}' is not registered in the entities database.")
                continue

            # Generate next sequential ID
            existing_ids = [int(p["id"][1:]) for p in promises_data["promises"] if p["id"].startswith('p')]
            next_id_num = max(existing_ids) + 1 if existing_ids else 1
            next_id = f"p{next_id_num:03d}"

            # Normalize deadline
            deadline_val = extracted_json.get("deadline_year", "ongoing")
            if not deadline_val or str(deadline_val).lower().strip() in ["null", "none", "n/a", ""]:
                deadline_val = "ongoing"

            # Normalize party
            party_val = extracted_json.get("party")
            if not party_val or str(party_val).lower().strip() in ["null", "none", "n/a", ""]:
                # Try parsing database party_str fallback
                if party_str:
                    try:
                        parties = json.loads(party_str)
                        if isinstance(parties, list) and len(parties) > 0:
                            party_val = parties[0]
                        else:
                            party_val = "Unknown Party"
                    except Exception:
                        party_val = "Unknown Party"
                else:
                    party_val = "Unknown Party"

            new_promise = {
                "id": next_id,
                "person": politician_name,
                "party": party_val,
                "role": "Politician",
                "promise": extracted_json.get("promise_text", title),
                "supporting_quote": extracted_json.get("supporting_quote", ""),
                "category": "general",
                "made_on": time.strftime("%Y-%m-%d", time.localtime(scraped_at)) if scraped_at else time.strftime("%Y-%m-%d"),
                "deadline": deadline_val,
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
