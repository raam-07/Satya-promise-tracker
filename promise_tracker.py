# ==============================================================================
# SATYA — PROMISE TRACKER (Repo 5)
#
# Reads promises.json + Classified Sheet and:
#   1. Autonomously extracts new promises from news articles (Zero-Human Mode)
#      - Dynamically loads tracked politicians from satya-entity-library
#   2. Links relevant articles to each promise as evidence (Delta Syncing)
#      - Score-based filtering (threshold 45)
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
from sentence_transformers import SentenceTransformer, util

# === HELPERS FOR PRECISION TRACKING ===

def parse_date_string(date_str):
    """
    Safely parses different ISO or custom date strings into a datetime object.
    Falls back to current time if parsing fails.
    """
    if not date_str:
        return datetime.now()
    date_str = str(date_str).strip()
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ', '%d-%m-%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    # Try custom extraction using regex for YYYY-MM-DD
    match = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', date_str)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    return datetime.now()

def extract_deadline_year(promise_text):
    """
    Extracts future or recent target years (e.g. 2024, 2025, 2026) from the promise text.
    Returns the integer year if found, otherwise None.
    """
    if not promise_text:
        return None
    years = re.findall(r'\b(202[0-9]|203[0-5])\b', promise_text)
    if years:
        return int(years[0])
    return None

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
CLASSIFIED_SHEET_NAME = 'Satya Classified'
CLASSIFIED_WORKSHEET_NAME = 'Sheet1'

PROMISES_JSON_URL = os.environ.get('PROMISES_JSON_URL', '')
ENTITIES_JSON_URL = os.environ.get('ENTITIES_JSON_URL', 'https://raw.githubusercontent.com/raam-07/satya-entity-library/main/entities.json')

PROMISES_OUTPUT_PATH = './promises.json'
REVIEW_PROMISES_PATH = './review_promises.json'

MODEL_PATH = os.environ.get('MODEL_PATH', './models/gemma-2-9b-it-Q6_K.gguf')

# Max articles to link per promise
MAX_EVIDENCE_ARTICLES = 10

# Minimum score to even consider an article for Gemma validation (Semantic similarity threshold)
MIN_SCORE_THRESHOLD = 45

# Max articles to send to Gemma for relevance check (saves time)
MAX_GEMMA_VALIDATION_BATCH = 3

# Maximum rows to process in a single execution to prevent timeouts
MAX_ROWS_PER_RUN = 500

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === LOAD SENTENCE TRANSFORMERS MODEL ===
try:
    logging.info("Loading Sentence Transformer model (all-MiniLM-L6-v2) for Pass 1...")
    encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence Transformer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Sentence Transformer model: {e}")
    encoder_model = None

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
Determine if the news article contains specific, factual information or updates regarding the progress, implementation, or roadblocks of this specific political promise.

Promise: "{promise_text}" made by {person}

CRITICAL RULES:
- The article MUST describe factual progress or updates about the exact action and target of the promise (e.g., if the promise is to 'build a bridge', the article must discuss progress or issues about building that specific bridge).
- Reject if the article only contains general background discussion or keywords without describing specific updates/progress related to this promise.

Article Title: {article_title}
Article Summary: {article_summary[:400]}

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

def gemma_assess_promise(llm, promise_text, person, evidence_texts, deadline_year=None):
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
    current_year = datetime.now().year

    deadline_context = ""
    if deadline_year:
        deadline_context = f"\nNote: The promise specifies a target deadline year of {deadline_year}. Today's year is {current_year}."

    prompt = f"""<start_of_turn>user
You are a factual promise fact-checker. Today's date is {current_date}. Based on the evidence articles below, assess the status of this political promise.{deadline_context}

Person: {person}
Promise: {promise_text}

Evidence from news articles:
{combined_evidence}

CRITICAL ASSESSMENT RULES:
- If a target deadline year ({deadline_year if deadline_year else 'N/A'}) has passed, and the evidence does not clearly confirm successful completion, mark the status as "broken".
- If the target deadline year has NOT passed, or if there is no deadline, and there is active ongoing work, mark the status as "ongoing".
- Mark status as "kept" ONLY if the evidence clearly confirms the promise is fully completed/delivered.

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

def find_politician_in_text(content, content_lower, minister_lookup):
    """
    Precisely and dynamically matches politicians in text without any hardcoded names or words.
    Uses surrounding capitalization, word boundaries, and allowed wordlists from entities.json.
    """
    sorted_aliases = sorted(minister_lookup.keys(), key=len, reverse=True)
    
    allowed_words_map = {}
    for alias_lower, canonical_name in minister_lookup.items():
        if canonical_name not in allowed_words_map:
            words = set()
            for w in canonical_name.lower().split():
                words.add(w)
            for a_lower, c_name in minister_lookup.items():
                if c_name == canonical_name:
                    for w in a_lower.split():
                        words.add(w)
            allowed_words_map[canonical_name] = words

    for alias in sorted_aliases:
        pattern = r'\b' + re.escape(alias) + r'\b'
        match = re.search(pattern, content_lower)
        
        if match:
            canonical_name = minister_lookup[alias]
            allowed_words = allowed_words_map.get(canonical_name, set())
            
            canonical_words = {w.lower() for w in canonical_name.split() if len(w) > 1}
            alias_words = set(alias.split())
            is_generic = not (alias_words & canonical_words)
            
            if is_generic:
                has_canonical = any(w in content_lower for w in canonical_words)
                if not has_canonical:
                    continue

            if ' ' not in alias:
                start_idx = match.start()
                end_idx = match.end()
                
                preceding_part = content[:start_idx].strip()
                preceding_words = re.findall(r'\b\w+\b', preceding_part)
                if preceding_words:
                    prec_word = preceding_words[-1]
                    if prec_word[0].isupper() and prec_word.lower() not in allowed_words:
                        continue
                
                succeeding_part = content[end_idx:].strip()
                succeeding_words = re.findall(r'\b\w+\b', succeeding_part)
                if succeeding_words:
                    succ_word = succeeding_words[0]
                    if succ_word[0].isupper() and succ_word.lower() not in allowed_words:
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

        matched_canonical, matched_keyword = find_politician_in_text(content, content_lower, minister_lookup)

        if not matched_canonical:
            continue

        logging.info(f"  [Article {index}] Found candidate for {matched_canonical} (via keyword '{matched_keyword}'): \"{title[:50]}...\"")
        logging.info("  [Gemma] Running promise extraction inference...")

        prompt = f"""<start_of_turn>user
You are a factual promise extractor. Analyze the news article below and determine if the politician ({matched_canonical}) has explicitly announced a concrete, measurable future policy promise or developmental target.

A valid promise MUST announce a concrete future target, developmental objective, or physical/digital/legislative deliverable (e.g., 'will build 50 schools', 'pledges to distribute laptops by 2026', 'will construct the dry port').

CRITICAL REJECTION RULES:
- Reject general administrative routines (e.g., 'will attend a meeting', 'will inspect a project', 'will travel to').
- Reject standard political statements, wishes, criticisms of other parties, or budget announcements that do not define clear, actionable future targets.
- Reject vague intentions without any specific deliverable.

Article Title: {title}
Article Summary: {summary[:400]}

Return ONLY a JSON response:
If a concrete policy promise matching the criteria above is identified:
{{
  "is_promise": true,
  "promise_text": "A clear, concise, single-sentence statement of the specific promise made (e.g. 'Committed to installing drinking water taps in all rural households.')",
  "category": "one of: farmer_agriculture, economy, infrastructure, corruption_scam, politics, crime_violence, education"
}}
If no concrete promise matching the strict criteria is identified:
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
                            "created_at": article.get('scraped_at', str(datetime.now().date())),
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
    Fallback method to build keywords if encoder is disabled.
    """
    keywords = set()
    person = promise.get('person', '')
    for part in person.lower().split():
        if len(part) > 4:
            keywords.add(part)
    promise_text = promise.get('promise', '').lower()
    for word in promise_text.split():
        word = re.sub(r'[^\w]', '', word)
        if len(word) > 4:
            keywords.add(word)
    category = promise.get('category', '')
    if category:
        for kw in category.split('_'):
            if len(kw) > 3:
                keywords.add(kw)
    return keywords

def score_article_for_promise(article, promise_keywords, person_name, promise):
    """
    Fallback score computation based on keywords.
    """
    content = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:400]}"
    text = content.lower()
    score = 0

    person_parts = person_name.lower().split()
    person_match = False
    
    allowed_words = {w.lower() for w in person_name.lower().split() if w}
    for kw in promise_keywords:
        for part in kw.split():
            allowed_words.add(part.lower())

    has_exclusion = False
    for part in person_parts:
        if len(part) > 3:
            pattern = r'\b' + re.escape(part) + r'\b'
            match = re.search(pattern, text)
            if match:
                start_idx = match.start()
                end_idx = match.end()
                
                preceding_part = content[:start_idx].strip()
                preceding_words = re.findall(r'\b\w+\b', preceding_part)
                if preceding_words:
                    prec_word = preceding_words[-1]
                    if prec_word[0].isupper() and prec_word.lower() not in allowed_words:
                        has_exclusion = True
                        break
                
                succeeding_part = content[end_idx:].strip()
                succeeding_words = re.findall(r'\b\w+\b', succeeding_part)
                if succeeding_words:
                    succ_word = succeeding_words[0]
                    if succ_word[0].isupper() and succ_word.lower() not in allowed_words:
                        has_exclusion = True
                        break

    if not has_exclusion:
        for part in person_parts:
            if len(part) > 3 and re.search(r'\b' + re.escape(part) + r'\b', text):
                score += 20
                person_match = True

    matched_keywords = 0
    for kw in promise_keywords:
        if kw.lower() in text:
            matched_keywords += 1
            score += 8

    if matched_keywords >= 2:
        score += 15
    if matched_keywords >= 4:
        score += 20

    if article.get('category') == promise.get('category', ''):
        score += 10

    if person_match and matched_keywords == 0:
        score = max(0, score - 25)

    return min(score, 100)

def link_articles_to_promises(promises_data, articles, llm):
    """
    For each promise, find the most relevant articles from the newly fetched batch.
    Two-pass approach:
    Pass 1: Semantic Embedding similarity OR Score-based pre-filter (fast, no AI)
    Pass 2: Gemma relevance validation (slow, high accuracy)
    """
    logging.info("--- Linking articles to promises (Semantic Match Mode) ---")

    # Build existing URL set per promise to avoid duplicates
    existing_per_promise = {}
    for promise in promises_data['promises']:
        existing_per_promise[promise['id']] = {
            a['url'] for a in promise.get('evidence_articles', [])
        }

    if not articles:
        return promises_data

    # Pre-encode all incoming articles (Title + Summary) to save compute
    article_embeddings = []
    valid_articles = []
    if encoder_model:
        logging.info("Pre-encoding article texts...")
        article_texts = []
        for article in articles:
            title = article.get('title', '')
            summary = article.get('rephrased_article', '')
            article_texts.append(f"{title} {summary[:400]}")
            valid_articles.append(article)
        
        if article_texts:
            article_embeddings = encoder_model.encode(article_texts, convert_to_tensor=True)
    else:
        valid_articles = articles

    for promise in promises_data['promises']:
        promise_id = promise['id']
        person = promise.get('person', '')
        promise_text = promise.get('promise', '')

        logging.info(f"Linking: [{promise_id}] {promise_text[:60]}...")

        # Resolve promise creation date for temporal matching
        promise_created_at_str = promise.get('created_at')
        if not promise_created_at_str:
            evidence = promise.get('evidence_articles', [])
            dates = [a.get('scraped_at') for a in evidence if a.get('scraped_at')]
            if dates:
                promise_created_at_str = min(dates)
            else:
                promise_created_at_str = str(datetime.now().date())
            promise['created_at'] = promise_created_at_str
        
        promise_created_at = parse_date_string(promise_created_at_str)

        scored_articles = []

        # --- PASS 1: Score-based filtering (Semantic Embeddings or Keyword Fallback) ---
        if encoder_model and len(article_embeddings) > 0:
            promise_embedding = encoder_model.encode(promise_text, convert_to_tensor=True)
            cos_scores = util.cos_sim(promise_embedding, article_embeddings)[0]
            
            for a_idx, article in enumerate(valid_articles):
                url = article.get('url', '')
                if url in existing_per_promise.get(promise_id, set()):
                    continue

                # TEMPORAL MATCHING: Reject evidence published before promise creation
                article_date_str = article.get('scraped_at')
                if article_date_str:
                    article_date = parse_date_string(article_date_str)
                    if article_date.date() < promise_created_at.date() - timedelta(days=1):
                        continue

                # Politician Match Check
                content = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:400]}"
                content_lower = content.lower()
                
                person_parts = person.lower().split()
                person_match = False
                
                allowed_words = {w.lower() for w in person.lower().split() if w}
                has_exclusion = False
                for part in person_parts:
                    if len(part) > 3:
                        pattern = r'\b' + re.escape(part) + r'\b'
                        match = re.search(pattern, content_lower)
                        if match:
                            start_idx = match.start()
                            end_idx = match.end()
                            
                            preceding_part = content[:start_idx].strip()
                            preceding_words = re.findall(r'\b\w+\b', preceding_part)
                            if preceding_words:
                                prec_word = preceding_words[-1]
                                if prec_word[0].isupper() and prec_word.lower() not in allowed_words:
                                    has_exclusion = True
                                    break
                            
                            succeeding_part = content[end_idx:].strip()
                            succeeding_words = re.findall(r'\b\w+\b', succeeding_part)
                            if succeeding_words:
                                succ_word = succeeding_words[0]
                                if succ_word[0].isupper() and succ_word.lower() not in allowed_words:
                                    has_exclusion = True
                                    break

                if not has_exclusion:
                    for part in person_parts:
                        if len(part) > 3 and re.search(r'\b' + re.escape(part) + r'\b', content_lower):
                            person_match = True
                            break

                if person_match:
                    similarity = cos_scores[a_idx].item()
                    score = int(similarity * 100)
                    
                    if article.get('category') == promise.get('category', ''):
                        score = min(score + 10, 100)
                        
                    if score >= MIN_SCORE_THRESHOLD:
                        scored_articles.append((score, article))
        else:
            # Fallback to pure keyword scoring
            keywords = build_promise_keywords(promise)
            for article in valid_articles:
                url = article.get('url', '')
                if url in existing_per_promise.get(promise_id, set()):
                    continue

                article_date_str = article.get('scraped_at')
                if article_date_str:
                    article_date = parse_date_string(article_date_str)
                    if article_date.date() < promise_created_at.date() - timedelta(days=1):
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

        deadline_year = extract_deadline_year(promise_text)

        suggested_status, reasoning, confidence = gemma_assess_promise(
            llm, promise_text, person, evidence_texts, deadline_year=deadline_year
        )

        if suggested_status:
            promise['gemma_suggestion'] = suggested_status
            promise['gemma_reasoning'] = reasoning
            promise['gemma_confidence'] = confidence
            promise['gemma_assessed_at'] = str(datetime.now().date())

            logging.info(f"[{promise['id']}] Current: {current_status} | Gemma: {suggested_status} ({confidence})")

            # Disagreement flag check with strictly high confidence gating for auto-updates
            if suggested_status != current_status:
                if confidence == 'high':
                    promise['status'] = suggested_status
                    logging.info(f"  → Automatically updated status to {suggested_status} based on {confidence} confidence")
                else:
                    logging.info(f"  → Gated auto-update: Gemma suggests {suggested_status} but confidence is only {confidence}. Keeping as {current_status}")
                
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
    logging.info("--- Satya Promise Tracker Started (Semantic/Autonomous Mode) ---")

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

    # 6. Autonomous extraction pass
    if articles and llm:
        if not minister_lookup:
            names = list({p.get('person') for p in promises_data['promises'] if p.get('person')})
            minister_lookup = {name.lower(): name for name in names}
            
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
