# ==============================================================================
# SATYA — PROMISE TRACKER (Repo 5)
#
# Reads promises.json + Classified Sheet and:
#   1. Links relevant articles to each promise as evidence
#   2. Uses Gemma to suggest status (kept/broken/ongoing) based on evidence
#   3. Generates review_promises.json for manual confirmation
#   4. Updates promises.json with confirmed evidence article links
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
PROMISES_OUTPUT_PATH = './promises.json'
REVIEW_PROMISES_PATH = './review_promises.json'

MODEL_PATH = "./models/gemma-2-2b-it-Q6_K_L.gguf"

# Max articles to link per promise
MAX_EVIDENCE_ARTICLES = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- GOOGLE SHEETS ---
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

def fetch_articles(sheet):
    logging.info("Fetching classified articles...")
    raw_data = sheet.col_values(1)
    articles = []
    for cell in raw_data:
        if not cell:
            continue
        try:
            articles.append(json.loads(cell))
        except json.JSONDecodeError:
            continue
    logging.info(f"Fetched {len(articles)} articles.")
    return articles

# ==============================================================================
# --- LOAD PROMISES ---
# ==============================================================================

def load_promises():
    if PROMISES_JSON_URL:
        try:
            logging.info("Fetching promises.json from GitHub...")
            response = requests.get(PROMISES_JSON_URL, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.warning(f"Failed to fetch from GitHub: {e}. Trying local.")

    if os.path.exists(PROMISES_OUTPUT_PATH):
        with open(PROMISES_OUTPUT_PATH, 'r') as f:
            return json.load(f)

    raise FileNotFoundError("No promises.json found.")

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
            n_ctx=2048,
            n_batch=256,
            n_threads=4,
            verbose=False
        )
        logging.info("Gemma loaded.")
        return llm
    except Exception as e:
        logging.error(f"Gemma failed to load: {e}")
        return None

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

    prompt = f"""<start_of_turn>user
You are a fact-checker. Based on the evidence articles below, assess the status of this political promise.

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

        return status, reasoning, confidence

    except Exception as e:
        logging.warning(f"Gemma promise assessment failed: {e}")
        return None, None, None

# ==============================================================================
# --- ARTICLE LINKING ---
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

    # Key words from promise text
    promise_text = promise.get('promise', '').lower()
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'will', 'that', 'this', 'is', 'are', 'was'}
    for word in promise_text.split():
        word = re.sub(r'[^\w]', '', word)
        if len(word) > 4 and word not in stop_words:
            keywords.add(word)

    # Category keywords
    category_keywords = {
        'farmer_agriculture': ['farmer', 'agriculture', 'kisan', 'crop', 'msp'],
        'economy': ['economy', 'gdp', 'jobs', 'employment', 'income', 'rupee'],
        'infrastructure': ['road', 'railway', 'hospital', 'school', 'electricity', 'water'],
        'corruption_scam': ['corruption', 'scam', 'black money', 'bribe'],
        'politics': ['election', 'parliament', 'government', 'policy'],
        'crime_violence': ['crime', 'police', 'security', 'violence'],
        'education': ['school', 'college', 'student', 'education'],
        'health': ['hospital', 'doctor', 'medicine', 'health'],
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

    # Person name match — highest weight
    person_parts = person_name.lower().split()
    for part in person_parts:
        if len(part) > 3 and part in text:
            score += 20

    # Keyword matches
    matched_keywords = 0
    for kw in promise_keywords:
        if kw in text:
            matched_keywords += 1
            score += 5

    # Bonus for multiple keyword matches
    if matched_keywords >= 3:
        score += 10
    if matched_keywords >= 5:
        score += 15

    # Category match
    if article.get('category') == promise.get('category', ''):
        score += 10

    # Skip pure international articles unless promise is about foreign policy
    if article.get('category') == 'international' and promise.get('category') != 'foreign_policy':
        score = max(0, score - 30)

    return min(score, 100)

def link_articles_to_promises(promises_data, articles):
    """
    For each promise, find the most relevant articles from classified sheet.
    Returns updated promises with evidence_articles filled.
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

        logging.info(f"Linking articles for: [{promise_id}] {promise_text[:60]}...")

        keywords = build_promise_keywords(promise)
        scored_articles = []

        for article in articles:
            url = article.get('url', '')
            if url in existing_per_promise.get(promise_id, set()):
                continue  # Already linked

            score = score_article_for_promise(article, keywords, person, promise)
            if score >= 30:  # Minimum relevance threshold
                scored_articles.append((score, article))

        # Sort by score, take top N
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        top_articles = scored_articles[:MAX_EVIDENCE_ARTICLES]

        new_links = []
        for score, article in top_articles:
            new_links.append({
                "url": article.get('url', ''),
                "title": article.get('title', ''),
                "source": article.get('source', ''),
                "scraped_at": article.get('scraped_at', ''),
                "relevance_score": score,
                "rephrased": article.get('rephrased_article', '')[:200]
            })

        if new_links:
            # Merge with existing evidence
            existing = promise.get('evidence_articles', [])
            all_urls = {a['url'] for a in existing}
            for link in new_links:
                if link['url'] not in all_urls:
                    existing.append(link)
                    all_urls.add(link['url'])

            # Keep top MAX_EVIDENCE_ARTICLES by relevance
            existing.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            promise['evidence_articles'] = existing[:MAX_EVIDENCE_ARTICLES]

            logging.info(f"  Linked {len(new_links)} new articles. Total evidence: {len(promise['evidence_articles'])}")
        else:
            logging.info(f"  No new articles found.")

    return promises_data

# ==============================================================================
# --- GEMMA STATUS ASSESSMENT ---
# ==============================================================================

def assess_promise_statuses(promises_data, llm):
    """
    For each promise with evidence articles, ask Gemma to suggest status.
    Populates gemma_suggestion and gemma_reasoning fields.
    """
    logging.info("--- Assessing promise statuses with Gemma ---")
    review_items = []

    for promise in promises_data['promises']:
        evidence = promise.get('evidence_articles', [])
        if not evidence:
            continue

        person = promise.get('person', '')
        promise_text = promise.get('promise', '')
        current_status = promise.get('status', 'ongoing')

        # Extract evidence texts
        evidence_texts = [a.get('rephrased', '') for a in evidence if a.get('rephrased')]

        if not evidence_texts:
            continue

        suggested_status, reasoning, confidence = gemma_assess_promise(
            llm, promise_text, person, evidence_texts
        )

        if suggested_status:
            promise['gemma_suggestion'] = suggested_status
            promise['gemma_reasoning'] = reasoning
            promise['gemma_confidence'] = confidence
            promise['gemma_assessed_at'] = str(datetime.now().date())

            logging.info(f"[{promise['id']}] Current: {current_status} | Gemma suggests: {suggested_status} ({confidence}) — {reasoning}")

            # Flag for review if Gemma disagrees with current status
            if suggested_status != current_status:
                review_items.append({
                    "promise_id": promise['id'],
                    "person": person,
                    "promise": promise_text[:100],
                    "current_status": current_status,
                    "gemma_suggestion": suggested_status,
                    "gemma_reasoning": reasoning,
                    "gemma_confidence": confidence,
                    "evidence_count": len(evidence),
                    "action_needed": "Review and update status in promises.json"
                })

    logging.info(f"Gemma disagrees on {len(review_items)} promises — flagged for review.")
    return promises_data, review_items

# ==============================================================================
# --- MAIN ---
# ==============================================================================

def main():
    start_time = time.time()
    logging.info("--- Satya Promise Tracker Started ---")

    # 1. Load data
    sheet = connect_to_sheets()
    articles = fetch_articles(sheet)
    promises_data = load_promises()

    logging.info(f"Loaded {len(promises_data['promises'])} promises.")

    # 2. Link articles to promises
    promises_data = link_articles_to_promises(promises_data, articles)

    # 3. Load Gemma and assess statuses
    llm = load_gemma()
    promises_data, review_items = assess_promise_statuses(promises_data, llm)

    # 4. Update metadata
    promises_data['metadata']['last_updated'] = str(datetime.now().date())
    promises_data['metadata']['total_promises'] = len(promises_data['promises'])
    promises_data['metadata']['promises_with_evidence'] = sum(
        1 for p in promises_data['promises'] if p.get('evidence_articles')
    )

    # 5. Save updated promises.json
    with open(PROMISES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(promises_data, f, indent=2, ensure_ascii=False)
    logging.info("Saved promises.json")

    # 6. Save review file
    review_output = {
        "generated_at": str(datetime.now()),
        "summary": {
            "total_promises": len(promises_data['promises']),
            "promises_with_evidence": promises_data['metadata']['promises_with_evidence'],
            "statuses_needing_review": len(review_items)
        },
        "review_items": review_items
    }

    with open(REVIEW_PROMISES_PATH, 'w', encoding='utf-8') as f:
        json.dump(review_output, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved review_promises.json ({len(review_items)} items need review)")

    elapsed = round(time.time() - start_time, 2)
    logging.info(f"--- Promise Tracker Finished in {elapsed}s ---")
    print(json.dumps(review_output['summary'], indent=2))

if __name__ == '__main__':
    main()
