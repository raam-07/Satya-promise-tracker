# ==============================================================================
# SATYA — PROMISE TRACKER (Repo 5)
#
# Reads promises.json + Classified Sheet and:
#   1. Links relevant articles to each promise as evidence
#      - Score-based filtering (threshold 70)
#      - Gemma pre-validation to confirm relevance before linking
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

MODEL_PATH = "./models/gemma-2-9b-it-Q6_K.gguf"

# Max articles to link per promise
MAX_EVIDENCE_ARTICLES = 10

# Minimum score to even consider an article for Gemma validation
MIN_SCORE_THRESHOLD = 55

# Max articles to send to Gemma for relevance check (saves time)
MAX_GEMMA_VALIDATION_BATCH = 20

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
            response = requests.get(PROMISES_JSON_URL.strip(), timeout=10)
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

    # Key words from promise text — STRICT: only highly specific words
    promise_text = promise.get('promise', '').lower()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'will', 'that', 'this', 'is', 'are', 'was',
        'every', 'year', 'india', 'indian', 'government', 'make', 'bring', 'end',
        'complete', 'build', 'crore', 'lakh'
    }
    for word in promise_text.split():
        word = re.sub(r'[^\w]', '', word)
        if len(word) > 5 and word not in stop_words:
            keywords.add(word)

    # Highly specific category keywords — only directly related terms
    category_keywords = {
        'farmer_agriculture': ['farmer', 'kisan', 'agricultural income', 'farm income', 'msp', 'crop price'],
        'economy': ['gdp', 'trillion', 'unemployment', 'job creation', 'employment generation', 'black money', 'swiss bank'],
        'infrastructure': ['bullet train', 'high speed rail', 'piped water', 'jal jeevan', 'pucca house', 'awas yojana'],
        'corruption_scam': ['electoral bond', 'corruption', 'transparency', 'political funding'],
        'politics': ['uniform civil code', 'ucc', 'one nation one election', 'rohingya'],
        'crime_violence': ['mafia', 'crime rate', 'encounter', 'gangster'],
        'education': ['government school', 'school quality', 'free education'],
    }
    for kw in category_keywords.get(promise.get('category', ''), []):
        keywords.add(kw)

    return keywords

def score_article_for_promise(article, promise_keywords, person_name, promise):
    """
    Score how relevant an article is to a promise.
    Returns a score 0-100.

    Stricter than before:
    - Person name alone is not enough
    - Must have specific keyword matches too
    """
    text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:400]}".lower()
    score = 0

    # Person name match
    person_parts = person_name.lower().split()
    person_match = False
    for part in person_parts:
        if len(part) > 3 and part in text:
            score += 20  # Restored to 20 to ensure high-signal matching
            person_match = True

    # Keyword matches — now much more important
    matched_keywords = 0
    matched_specific = []
    for kw in promise_keywords:
        if kw.lower() in text:
            matched_keywords += 1
            matched_specific.append(kw)
            score += 8  # Increased keyword weight

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
    For each promise, find the most relevant articles from classified sheet.

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
                "content": article.get('content', '')  # <--- ADD THIS LINE
            })

        if new_links:
            existing = promise.get('evidence_articles', [])
            # Remove old unvalidated articles (those without gemma_validated flag)
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
            # Clear old unvalidated evidence if we found no new valid articles
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
    Ignores low confidence Gemma suggestions when current status is manually set.
    """
    logging.info("--- Assessing promise statuses with Gemma ---")
    review_items = []

    for promise in promises_data['promises']:
        evidence = promise.get('evidence_articles', [])

        # Only assess if we have Gemma-validated evidence
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
            promise['gemma_suggestion'] = suggested_status
            promise['gemma_reasoning'] = reasoning
            promise['gemma_confidence'] = confidence
            promise['gemma_assessed_at'] = str(datetime.now().date())

            logging.info(f"[{promise['id']}] Current: {current_status} | Gemma: {suggested_status} ({confidence}) — {reasoning}")

            # Only flag for review if Gemma disagrees AND has medium/high confidence
            # Low confidence disagreements are too unreliable to surface
            if suggested_status != current_status and confidence in ['medium', 'high']:
                review_items.append({
                    "promise_id": promise['id'],
                    "person": person,
                    "promise": promise_text[:100],
                    "current_status": current_status,
                    "gemma_suggestion": suggested_status,
                    "gemma_reasoning": reasoning,
                    "gemma_confidence": confidence,
                    "evidence_count": len(validated_evidence),
                    "action_needed": "Review and update status in promises.json"
                })
            elif suggested_status != current_status and confidence == 'low':
                logging.info(f"  → Gemma disagrees but low confidence — ignoring suggestion")

    logging.info(f"Gemma disagrees (medium/high confidence) on {len(review_items)} promises.")
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

    # 2. Load Gemma early — needed for both linking and assessment
    llm = load_gemma()

    # 3. Link articles to promises (with Gemma validation)
    promises_data = link_articles_to_promises(promises_data, articles, llm)

    # 4. Assess promise statuses with Gemma
    promises_data, review_items = assess_promise_statuses(promises_data, llm)

    # 5. Update metadata
    promises_data['metadata']['last_updated'] = str(datetime.now().date())
    promises_data['metadata']['total_promises'] = len(promises_data['promises'])
    promises_data['metadata']['promises_with_evidence'] = sum(
        1 for p in promises_data['promises']
        if any(a.get('gemma_validated') for a in p.get('evidence_articles', []))
    )

    # 6. Save updated promises.json
    with open(PROMISES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(promises_data, f, indent=2, ensure_ascii=False)
    logging.info("Saved promises.json")

    # 7. Save review file
    review_output = {
        "generated_at": str(datetime.now()),
        "summary": {
            "total_promises": len(promises_data['promises']),
            "promises_with_validated_evidence": promises_data['metadata']['promises_with_evidence'],
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
