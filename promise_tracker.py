# ==============================================================================
# SATYA — PROMISE TRACKER (Repo 5)
#
# Reads promises.json + Classified Sheet and:
#   1. Autonomously extracts new promises from news articles (Zero-Human Mode)
#      - Dynamically loads tracked politicians from satya-entity-library
#   2. Links relevant articles to each promise as evidence (Delta Syncing)
#      - Score-based filtering (threshold 55)
#      - Gemma pre-validation to confirm relevance before linking
#   3. Uses Gemma to suggest status (kept/broken/ongoing) based on evidence
#   4. Updates promises.json with newly discovered promises and evidence
#
# Runs weekly via GitHub Actions.
# ==============================================================================

import os
import json
import time
import logging
import re
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
CLASSIFIED_SHEET_NAME = 'Satya Classified'
CLASSIFIED_WORKSHEET_NAME = 'Sheet1'

PROMISES_JSON_URL = os.environ.get('PROMISES_JSON_URL', '')
ENTITIES_JSON_URL = os.environ.get('ENTITIES_JSON_URL', 'https://raw.githubusercontent.com/raam-07/satya-entity-library/main/entities.json')

PROMISES_OUTPUT_PATH = './promises.json'
REVIEW_PROMISES_PATH = './review_promises.json'

MODEL_PATH = "./models/gemma-2-9b-it-Q6_K.gguf"

# Max articles to link per promise
MAX_EVIDENCE_ARTICLES = 10

# Minimum score to even consider an article for Gemma validation
MIN_SCORE_THRESHOLD = 55

# Max articles to send to Gemma for relevance check (saves time)
MAX_GEMMA_VALIDATION_BATCH = 3

# Maximum rows to process in a single execution to prevent timeouts
MAX_ROWS_PER_RUN = 150

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- GOOGLE SHEETS & DELTA SYNCING ---
# ==============================================================================

def connect_to_sheets():
    logging.info("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not gcp_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON missing!")
    creds_dict = json.loads(gcp_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(CLASSIFIED_SHEET_NAME).worksheet(CLASSIFIED_WORKSHEET_NAME)
    logging.info("Connected.")
    return sheet

def fetch_new_articles(sheet, start_row=1):
    """
    Downloads classified articles in a single batch, parsing only new rows 
    starting from start_row index up to MAX_ROWS_PER_RUN limit to avoid timeouts.
    """
    logging.info(f"Fetching classified articles starting from row {start_row} (max cap: {MAX_ROWS_PER_RUN})...")
    all_rows = sheet.get_all_values()
    total_rows = len(all_rows)
    
    # Calculate the end index for our slice
    end_row = min(start_row + MAX_ROWS_PER_RUN, total_rows)
    
    articles = []
    # Process only rows within the start_row to end_row slice
    for index, row in enumerate(all_rows[start_row:end_row], start=start_row + 1):
        if not row or not row[0]:
            continue
        try:
            article = json.loads(row[0])
            article['sheet_row'] = index
            
            # LIVE LOGGING: Real-time progress on ingestion
            logging.info(f"  [Row {index}/{total_rows}] Ingested: \"{article.get('title', '')[:50]}...\" from {article.get('source', '')}")
            
            articles.append(article)
        except json.JSONDecodeError:
            continue
            
    logging.info(f"Fetched {len(articles)} new articles. Processed up to row {end_row} of {total_rows}.")
    return articles, end_row

# ==============================================================================
# --- LOAD PROMISES & ENTITIES ---
# ==============================================================================

def load_promises():
    if PROMISES_JSON_URL:
        try:
            logging.info("Fetching promises.json from GitHub...")
            response = requests.get(PROMISES_JSON_URL.strip(), timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.warning(f"Failed to fetch from GitHub: {e}. Trying local.")

    if os.path.exists(PROMISES_OUTPUT_PATH):
        with open(PROMISES_OUTPUT_PATH, 'r') as f:
            return json.load(f)

    # Initialize a clean structure if not found
    logging.info("Creating a fresh promises structure...")
    return {
        "metadata": {
            "last_updated": str(datetime.now().date()),
            "total_promises": 0,
            "promises_with_evidence": 0,
            "last_processed_row": 1
        },
        "promises": []
    }

def load_tracked_politicians():
    """
    Dynamically loads the master list of all politicians from your entities.json database.
    """
    url = ENTITIES_JSON_URL.strip()
    try:
        logging.info("Fetching master entities.json to load politicians...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        entities = response.json()
        
        # Extract names from cabinet_ministers, state_chief_ministers, and opposition_leaders
        all_ministers = (
            entities.get('india', {}).get('cabinet_ministers', []) +
            entities.get('india', {}).get('state_chief_ministers', []) +
            entities.get('india', {}).get('opposition_leaders', [])
        )
        
        # Capture canonical names and aliases
        lookup = {}
        for m in all_ministers:
            name = m.get('name')
            if name:
                lookup[name.lower()] = name
                for alias in m.get('aliases', []):
                    if alias:
                        lookup[alias.lower()] = name
                        
        logging.info(f"Loaded {len(lookup)} unique politician names/aliases dynamically from entities.json")
        return lookup
    except Exception as e:
        logging.warning(f"Failed to load entities.json dynamically: {e}. Falling back to promises.json names.")
        return {}

# ==============================================================================
# --- GEMMA ---
# ==============================================================================

def load_gemma():
    try:
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        logging.info("Loading Gemma...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_batch=512,
            n_threads=2,
            verbose=False
        )
        logging.info("Gemma loaded.")
        return llm
    except Exception as e:
        logging.error(f"Gemma failed to load: {e}")
        return None

def gemma_is_article_relevant(llm, promise_text, person, article_title, article_summary):
    """
    Ask Gemma: does this article contain specific information about this promise?
    Returns True/False.
    Fast check — very short prompt, yes/no answer only.
    """
    if llm is None:
        return True  # If no Gemma, let it through

    prompt = f"""<start_of_turn>user
Does the article below contain specific factual information about this political promise?

Promise: "{promise_text}" made by {person}

Article Title: {article_title}
Article Summary: {article_summary[:300]}

Answer with ONLY a JSON: {{"relevant": "yes" or "no"}}
No explanation. No extra text.
<end_of_turn>
<start_of_turn>model
"""
    try:
        start_time = time.time()
        response = llm(
            prompt,
            max_tokens=20,
            temperature=0.1,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )
        raw = response['choices'][0].get('text', '').strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        elapsed = round(time.time() - start_time, 2)
        logging.info(f"    [Gemma] Relevance check completed in {elapsed}s.")
        return parsed.get('relevant', 'no').lower() == 'yes'
    except Exception:
        return False  # If Gemma fails, reject the article (safe default)

def gemma_assess_promise(llm, promise_text, person, evidence_texts):
    """
    Given a promise and evidence articles, Gemma suggests:
    - status: kept / broken / ongoing
    - reasoning: one sentence explanation
    - confidence: high / medium / low
    """
    if llm is None or not evidence_texts:
        return None, None, None

    combined_evidence = " | ".join(evidence_texts[:3])[:800]

    current_date = datetime.now().strftime("%B %d, %Y")
    prompt = f"""<start_of_turn>user
You are a fact-checker. Today's date is {current_date}. Based on the evidence articles below, assess the status of this political promise.

Person: {person}
Promise: {promise_text}

Evidence from news articles:
{combined_evidence}

Return ONLY a JSON object with these fields:
- "status": one of "kept", "broken", "ongoing"
- "reasoning": one sentence explaining why (max 30 words)
- "confidence": one of "high", "medium", "low"

No explanation. No extra text. Only JSON.
<end_of_turn>
<start_of_turn>model
"""
    try:
        start_time = time.time()
        response = llm(
            prompt,
            max_tokens=150,
            temperature=0.1,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )
        raw = response['choices'][0].get('text', '').strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)

        status = parsed.get('status', 'ongoing').lower()
        if status not in ['kept', 'broken', 'ongoing']:
            status = 'ongoing'

        reasoning = str(parsed.get('reasoning', '')).strip()
        confidence = parsed.get('confidence', 'low').lower()
        elapsed = round(time.time() - start_time, 2)
        logging.info(f"    [Gemma] Status assessment completed in {elapsed}s.")

        return status, reasoning, confidence

    except Exception as e:
        logging.warning(f"Gemma promise assessment failed: {e}")
        return None, None, None

# ==============================================================================
# --- AUTONOMOUS PROMISE EXTRACTION ---
# ==============================================================================

def find_politician_in_text(content_lower, minister_lookup):
    """
    Precisely matches politicians in text using word boundaries and name-specific exclusions.
    Returns (canonical_name, matched_alias) or (None, None).
    """
    # Sort aliases by length descending to match longer specific aliases first
    sorted_aliases = sorted(minister_lookup.keys(), key=len, reverse=True)
    
    for alias in sorted_aliases:
        # Match as whole word only to prevent matching "modi" in "modifies" or "ak" in "Pakistan"
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, content_lower):
            canonical_name = minister_lookup[alias]
            
            # --- SPECIFIC EXCLUSIONS FOR HIGH-FALSE-POSITIVE ALIASES ---
            if alias == 'shah':
                # Exclude Pakistani cricketer Naseem Shah, Sindh CM Murad Ali Shah, Shah Rukh Khan, etc.
                if any(x in content_lower for x in ['naseem shah', 'murad ali shah', 'shah rukh', 'sher shah', 'danial shah']):
                    continue
            
            elif alias == 'arvind':
                # Exclude economist Arvind Panagariya, Arvind Subramanian, etc.
                if any(x in content_lower for x in ['arvind panagariya', 'arvind subramanian', 'arvind gupta']):
                    continue
            
            elif alias == 'modi':
                # Exclude Lalit Modi, Nirav Modi, Sushil Modi (unless it's PM Modi)
                if any(x in content_lower for x in ['nirav modi', 'lalit modi', 'sushil modi']) and not any(x in content_lower for x in ['narendra modi', 'pm modi', 'modiji']):
                    continue

            return canonical_name, alias
            
    return None, None

def extract_new_promises_from_articles(llm, articles, existing_promises, minister_lookup):
    """
    Evaluates new articles and uses Gemma to discover and extract new concrete policy promises.
    Uses dynamic politician list from entities.json as a fast Python filter for 100% precision.
    """
    if llm is None or not articles:
        return []

    logging.info("--- Running Promise Extraction Pass (Dynamic Entities) ---")
    new_extracted_promises = []
    existing_text_list = [p['promise'].lower() for p in existing_promises]

    for index, article in enumerate(articles, start=1):
        title = article.get('title', '')
        summary = article.get('rephrased_article', '')
        content = f"{title} {summary}"
        content_lower = content.lower()

        # 1. Match against master dynamic politician list using high-precision regex and name exclusions
        matched_canonical, matched_keyword = find_politician_in_text(content_lower, minister_lookup)

        if not matched_canonical:
            continue

        # LIVE LOGGING: Show active candidate matching and gemma start
        logging.info(f"  [Article {index}] Found candidate for {matched_canonical} (via keyword '{matched_keyword}'): \"{title[:50]}...\"")
        logging.info("  [Gemma] Running promise extraction inference...")

        # 2. Ask Gemma if a concrete promise has been announced in this news
        prompt = f"""<start_of_turn>user
Analyze the news article below. Determine if the politician ({matched_canonical}) has explicitly made a concrete, measurable future policy promise or developmental target (e.g., "will build X by Y", "pledges to provide Z"). 

Do not include routine political statements, criticisms of the opposition, general administrative duties, or scheduling announcements.

Article Title: {title}
Article Summary: {summary[:400]}

Return ONLY a JSON response:
If a concrete policy promise is identified:
{{
  "is_promise": true,
  "promise_text": "A clear, concise, single-sentence statement of the specific promise made (e.g. 'Committed to installing drinking water taps in all rural households.')",
  "category": "one of: farmer_agriculture, economy, infrastructure, corruption_scam, politics, crime_violence, education"
}}
If no concrete promise is identified:
{{
  "is_promise": false
}}
No explanation. No extra text. Only JSON.
<end_of_turn>
<start_of_turn>model
"""
        try:
            start_inference = time.time()
            response = llm(
                prompt,
                max_tokens=150,
                temperature=0.1,
                stop=["<end_of_turn>", "<start_of_turn>"],
                echo=False
            )
            raw = response['choices'][0].get('text', '').strip()
            raw = re.sub(r'```json|```', '', raw).strip()
            parsed = json.loads(raw)
            elapsed = round(time.time() - start_inference, 2)
            logging.info(f"    [Gemma] Inference finished in {elapsed}s.")

            if parsed.get('is_promise') is True:
                promise_candidate = parsed.get('promise_text', '').strip()
                category = parsed.get('category', 'politics')

                if promise_candidate and category in ['farmer_agriculture', 'economy', 'infrastructure', 'corruption_scam', 'politics', 'crime_violence', 'education']:
                    # 3. De-duplication check against existing promises
                    is_duplicate = False
                    words_candidate = set(re.findall(r'\w+', promise_candidate.lower()))
                    
                    for existing in existing_text_list:
                        words_existing = set(re.findall(r'\w+', existing))
                        common_words = words_candidate & words_existing
                        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'will', 'that', 'this', 'is', 'are', 'was'}
                        important_overlap = common_words - stop_words
                        if len(important_overlap) > 5:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        new_promise_entry = {
                            "id": f"promise_{int(time.time())}_{len(new_extracted_promises)}",
                            "person": matched_canonical,
                            "promise": promise_candidate,
                            "category": category,
                            "status": "ongoing",
                            "evidence_articles": []
                        }
                        new_extracted_promises.append(new_promise_entry)
                        existing_text_list.append(promise_candidate.lower())
                        logging.info(f"    ★ SUCCESS: Extracted New Promise: [{matched_canonical}] {promise_candidate[:60]}...")
            else:
                logging.info("    [Gemma] Rejected: No concrete promise found in this news article.")
        except Exception as e:
            logging.warning(f"    Failed to parse new promise from article: {e}")
            continue

    logging.info(f"Auto-extracted {len(new_extracted_promises)} new promises from the incoming articles.")
    return new_extracted_promises

# ==============================================================================
# --- ARTICLE LINKING & SCORING ---
# ==============================================================================

def build_promise_keywords(promise):
    """
    Build a keyword set for matching articles to a promise.
    Combines person name, promise text keywords, and category.
    """
    keywords = set()

    # Person name and aliases
    person = promise.get('person', '')
    for part in person.lower().split():
        if len(part) > 3:
            keywords.add(part)

    # Key words from promise text — relaxed length limit to capture high-signal terms
    promise_text = promise.get('promise', '').lower()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'will', 'that', 'this', 'is', 'are', 'was',
        'every', 'year', 'india', 'indian', 'government', 'make', 'bring', 'end',
        'complete', 'build', 'crore', 'lakh', 'were', 'have', 'been', 'their', 
        'they', 'them', 'more', 'about', 'would', 'should', 'after', 'under'
    }
    for word in promise_text.split():
        word = re.sub(r'[^\w]', '', word)
        if len(word) > 3 and word not in stop_words: # Changed from > 5 to > 3
            keywords.add(word)

    # Highly specific category keywords
    category_keywords = {
        'farmer_agriculture': ['farmer', 'kisan', 'agricultural', 'msp', 'crop', 'agriculture'],
        'economy': ['gdp', 'trillion', 'unemployment', 'job', 'employment', 'money', 'bank', 'tax'],
        'infrastructure': ['train', 'rail', 'water', 'road', 'house', 'yojana', 'power', 'highway'],
        'corruption_scam': ['corruption', 'scam', 'transparency', 'bond'],
        'politics': ['ucc', 'election', 'nation', 'rohingya'],
        'crime_violence': ['mafia', 'crime', 'encounter', 'gangster', 'violence'],
        'education': ['school', 'education', 'college', 'teacher'],
    }
    for kw in category_keywords.get(promise.get('category', ''), []):
        keywords.add(kw)

    return keywords

def score_article_for_promise(article, promise_keywords, person_name, promise):
    """
    Score how relevant an article is to a promise.
    Returns a score 0-100.
    """
    text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:400]}".lower()
    score = 0

    # Person name match
    person_parts = person_name.lower().split()
    person_match = False
    
    # Exclusions for matching name parts in this specific article
    has_exclusion = False
    p_name_lower = person_name.lower()
    if p_name_lower == 'amit shah':
        if any(x in text for x in ['naseem shah', 'murad ali shah', 'shah rukh', 'sher shah', 'danial shah']):
            has_exclusion = True
    elif p_name_lower == 'arvind kejriwal':
        if any(x in text for x in ['arvind panagariya', 'arvind subramanian', 'arvind gupta']):
            has_exclusion = True
    elif p_name_lower == 'narendra modi':
        if any(x in text for x in ['nirav modi', 'lalit modi', 'sushil modi']) and not any(x in text for x in ['narendra modi', 'pm modi', 'modiji']):
            has_exclusion = True

    if not has_exclusion:
        for part in person_parts:
            if len(part) > 3 and re.search(r'\b' + re.escape(part) + r'\b', text):
                score += 20
                person_match = True

    # Keyword matches
    matched_keywords = 0
    for kw in promise_keywords:
        if kw.lower() in text:
            matched_keywords += 1
            score += 8

    # Bonus for multiple keyword matches
    if matched_keywords >= 2:
        score += 15
    if matched_keywords >= 4:
        score += 20

    # Category match bonus
    if article.get('category') == promise.get('category', ''):
        score += 10

    # PENALTY: If person mentioned but zero specific keywords → likely irrelevant
    if person_match and matched_keywords == 0:
        score = max(0, score - 25)

    # PENALTY: International articles for domestic promises
    if article.get('category') == 'international' and promise.get('category') != 'foreign_policy':
        score = max(0, score - 40)

    # PENALTY: Stock market / finance articles for non-economy promises
    if promise.get('category') != 'economy':
        if article.get('source') == 'Economic Times' and 'stock' in text:
            score = max(0, score - 20)

    return min(score, 100)

def link_articles_to_promises(promises_data, articles, llm):
    """
    For each promise, find the most relevant articles from the newly fetched batch.
    Two-pass approach:
    Pass 1: Score-based filtering (fast, no AI)
    Pass 2: Gemma relevance validation (slow, high accuracy)
    """
    logging.info("--- Linking articles to promises ---")

    # Build existing URL set per promise to avoid duplicates
    existing_per_promise = {}
    for promise in promises_data['promises']:
        existing_per_promise[promise['id']] = {
            a['url'] for a in promise.get('evidence_articles', [])
        }

    for promise in promises_data['promises']:
        promise_id = promise['id']
        person = promise.get('person', '')
        promise_text = promise.get('promise', '')

        logging.info(f"Linking: [{promise_id}] {promise_text[:60]}...")

        keywords = build_promise_keywords(promise)
        scored_articles = []

        # --- PASS 1: Score-based filtering ---
        for article in articles:
            url = article.get('url', '')
            if url in existing_per_promise.get(promise_id, set()):
                continue

            score = score_article_for_promise(article, keywords, person, promise)
            if score >= MIN_SCORE_THRESHOLD:
                scored_articles.append((score, article))

        # Sort by score, take top batch for Gemma validation
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        candidates = scored_articles[:MAX_GEMMA_VALIDATION_BATCH]

        logging.info(f"  Pass 1: {len(candidates)} candidates above threshold {MIN_SCORE_THRESHOLD}")

        # --- PASS 2: Gemma relevance validation ---
        validated_articles = []
        for score, article in candidates:
            title = article.get('title', '')
            summary = article.get('rephrased_article', '')

            is_relevant = gemma_is_article_relevant(
                llm, promise_text, person, title, summary
            )

            if is_relevant:
                validated_articles.append((score, article))
                logging.info(f"  ✓ Gemma APPROVED: {title[:60]}")
            else:
                logging.info(f"  ✗ Gemma REJECTED: {title[:60]}")

        logging.info(f"  Pass 2: {len(validated_articles)} articles validated by Gemma")

        # Take top MAX_EVIDENCE_ARTICLES
        top_articles = validated_articles[:MAX_EVIDENCE_ARTICLES]

        new_links = []
        for score, article in top_articles:
            new_links.append({
                "url": article.get('url', ''),
                "title": article.get('title', ''),
                "source": article.get('source', ''),
                "scraped_at": article.get('scraped_at', ''),
                "relevance_score": score,
                "gemma_validated": True,
                "rephrased": article.get('rephrased_article', '')[:200],
                "content": article.get('content', '')
            })

        if new_links:
            existing = promise.get('evidence_articles', [])
            # Remove old unvalidated articles
            existing = [a for a in existing if a.get('gemma_validated', False)]
            all_urls = {a['url'] for a in existing}

            for link in new_links:
                if link['url'] not in all_urls:
                    existing.append(link)
                    all_urls.add(link['url'])

            existing.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            promise['evidence_articles'] = existing[:MAX_EVIDENCE_ARTICLES]

            logging.info(f"  Saved {len(new_links)} validated articles. Total: {len(promise['evidence_articles'])}")
        else:
            old_evidence = promise.get('evidence_articles', [])
            promise['evidence_articles'] = [a for a in old_evidence if a.get('gemma_validated', False)]
            logging.info(f"  No new validated articles found.")

    return promises_data

# ==============================================================================
# --- GEMMA STATUS ASSESSMENT ---
# ==============================================================================

def assess_promise_statuses(promises_data, llm):
    """
    For each promise with validated evidence, ask Gemma to suggest status.
    Only runs on promises that have gemma_validated evidence articles.
    """
    logging.info("--- Assessing promise statuses with Gemma ---")
    review_items = []

    for promise in promises_data['promises']:
        evidence = promise.get('evidence_articles', [])

        validated_evidence = [a for a in evidence if a.get('gemma_validated', False)]
        if not validated_evidence:
            logging.info(f"[{promise['id']}] Skipping — no validated evidence articles")
            continue

        person = promise.get('person', '')
        promise_text = promise.get('promise', '')
        current_status = promise.get('status', 'ongoing')

        evidence_texts = [a.get('rephrased', '') for a in validated_evidence if a.get('rephrased')]

        if not evidence_texts:
            continue

        suggested_status, reasoning, confidence = gemma_assess_promise(
            llm, promise_text, person, evidence_texts
        )

        if suggested_status:
            # Save the suggestion metadata inside the promise itself
            promise['gemma_suggestion'] = suggested_status
            promise['gemma_reasoning'] = reasoning
            promise['gemma_confidence'] = confidence
            promise['gemma_assessed_at'] = str(datetime.now().date())

            logging.info(f"[{promise['id']}] Current: {current_status} | Gemma: {suggested_status} ({confidence})")

            # Disagreement flag check (for manual reviews or audit lists if needed)
            if suggested_status != current_status and confidence in ['medium', 'high']:
                # Update status directly if we are in fully automated mode (can be toggled)
                promise['status'] = suggested_status
                logging.info(f"  → Automatically updated status to {suggested_status} based on {confidence} confidence")
                
                review_items.append({
                    "promise_id": promise['id'],
                    "person": person,
                    "promise": promise_text[:100],
                    "previous_status": current_status,
                    "new_status": suggested_status,
                    "gemma_reasoning": reasoning,
                    "gemma_confidence": confidence,
                    "evidence_count": len(validated_evidence)
                })

    return promises_data, review_items

# ==============================================================================
# --- MAIN ---
# ==============================================================================

def main():
    start_time = time.time()
    logging.info("--- Satya Promise Tracker Started (Autonomous/Delta Mode) ---")

    # 1. Load active promise registry
    promises_data = load_promises()
    logging.info(f"Loaded {len(promises_data['promises'])} existing promises.")

    # 2. Extract last processed row index from metadata (defaulting to 1 if first run)
    if 'metadata' not in promises_data:
        promises_data['metadata'] = {}
    last_processed_row = promises_data['metadata'].get('last_processed_row', 1)

    # 3. Connect to sheets and ingest delta (only new rows)
    sheet = connect_to_sheets()
    articles, newest_row = fetch_new_articles(sheet, start_row=last_processed_row)

    # 4. Load master politician lookup from entities.json
    minister_lookup = load_tracked_politicians()

    # 5. Load Gemma inference engine
    llm = load_gemma()

    # 6. Autonomous extraction pass (only if there are new articles and lookup loaded successfully)
    if articles and llm:
        # Fallback to promises names if entities.json could not be loaded
        if not minister_lookup:
            names = list({p.get('person') for p in promises_data['promises'] if p.get('person')})
            minister_lookup = {name.lower(): name for name in names}
            
        # Discover and extract new promises from the incoming stream (Dynamic/Optimized Pass)
        new_promises = extract_new_promises_from_articles(llm, articles, promises_data['promises'], minister_lookup)
        if new_promises:
            promises_data['promises'].extend(new_promises)
            logging.info(f"Auto-added {len(new_promises)} new promises directly to database.")

    # 7. Link articles to promises as evidence
    if articles:
        promises_data = link_articles_to_promises(promises_data, articles, llm)

    # 8. Evaluate and assess promise statuses
    promises_data, review_items = assess_promise_statuses(promises_data, llm)

    # 9. Update execution metadata
    promises_data['metadata']['last_updated'] = str(datetime.now().date())
    promises_data['metadata']['total_promises'] = len(promises_data['promises'])
    promises_data['metadata']['last_processed_row'] = newest_row
    promises_data['metadata']['promises_with_evidence'] = sum(
        1 for p in promises_data['promises']
        if any(a.get('gemma_validated') for a in p.get('evidence_articles', []))
    )

    # 10. Save updated promises.json
    with open(PROMISES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(promises_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved promises.json with updated last_processed_row = {newest_row}")

    # 11. Save review file (retains audit compatibility for status shifts)
    review_output = {
        "generated_at": str(datetime.now()),
        "summary": {
            "total_promises": len(promises_data['promises']),
            "promises_with_validated_evidence": promises_data['metadata']['promises_with_evidence'],
            "statuses_automatically_updated": len(review_items)
        },
        "updated_items": review_items
    }

    with open(REVIEW_PROMISES_PATH, 'w', encoding='utf-8') as f:
        json.dump(review_output, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved review_promises.json ({len(review_items)} updates audited)")

    elapsed = round(time.time() - start_time, 2)
    logging.info(f"--- Promise Tracker Finished in {elapsed}s ---")
    print(json.dumps(review_output['summary'], indent=2))

if __name__ == '__main__':
    main()
