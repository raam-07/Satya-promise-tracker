# Satya Promise Tracker — Improvement Roadmap & Integrity Backlog

This document outlines key architectural and data integrity issues identified in the political promise tracking module, along with their suggested fixes. These enhancements are prioritized by their impact on the platform's credibility and data quality.

---

## 🚨 High Priority (Platform Credibility & Data Integrity)

### 1. Verdict Instability & Lack of Confidence Gating
* **Issue**: Verdicts (e.g. Kept, Broken, Ongoing) can flip based on a single news article, and then flip back on the next. There is no confidence gating; any matching article overwrites a promise's status, and confidence is hardcoded to "high."
* **Suggested Fix**: 
  - Only auto-apply status changes when the model outputs a genuinely high-confidence assessment.
  - Require at least **2 independent news sources** before publishing a critical status change like a `"broken"` verdict.
  - Any status change that doesn't meet these requirements must be left at the current status and written to a manual review queue file (`review_promises.json`).

### 2. Accumulation of Duplicate Promises
* **Issue**: Whether a promise is a new entry or an update to an existing one is decided entirely by the LLM outputting a matched ID. When the LLM fails to match (hallucinations or context limits), it re-creates a promise that is already tracked, causing duplicates to accumulate.
* **Suggested Fix**:
  - Implement a programmatic text-similarity check (e.g., TF-IDF or token-based overlap) against existing promises for the same politician *before* appending a new promise.
  - If similarity is high, automatically merge/treat as an update.
  - Borderline cases should be routed to `review_promises.json` for manual resolution.

---

## 📈 Medium Priority (Quality & Structural Drift)

### 3. All New Promises Default to Category "general"
* **Issue**: The Stage 2 Extractor does not extract a category field. As a result, all newly scraped promises default to `"general"`, which degrades dashboard classification quality as the database grows.
* **Suggested Fix**:
  - Add a `category` field to the Stage 2 extraction schema.
  - Enforce the extraction to map to the project's canonical category list (e.g., `farmer_agriculture`, `economy`, `health`, `infrastructure`, etc.).
  - Fall back to `"general"` only if no canonical categories apply.

### 4. Infinite Self-Loop Risk on Write Failures
* **Issue**: If the database update marking articles as `'processed'` fails, the pipeline will re-fetch the same batch of articles in the next runner run, causing an infinite loop that wastes LLM calls.
* **Suggested Fix**:
  - Add a check at the end of each run to confirm that the database row pointer (`last_processed_row`) has actually advanced.
  - If it hasn't advanced, automatically terminate the workflow loop and raise a GitHub Actions alert instead of triggering the next run.

### 5. Lack of Verdict History/Trail
* **Issue**: The promise status is only ever "what the last processed article said." A promise that was previously marked as "broken" can silently revert to "ongoing" without leaving any history or audit trail.
* **Suggested Fix**:
  - Implement an append-only status trajectory log per promise (e.g., `status_history: [{"date": "YYYY-MM-DD", "status": "broken", "evidence_url": "..."}]`) so users can audit how a verdict evolved over time.

### 6. Misaligned `made_on` Timestamps
* **Issue**: The pipeline uses the article's scraping date for `made_on`, rather than when the promise was actually declared. Consequently, older promises reported today appear in the tracker as if they were made today.
* **Suggested Fix**:
  - Ask the LLM to extract the date the promise was made from the article text.
  - If the exact date is unclear or missing, label it as `"reported_on"` instead of `"made_on"`.

---

## 🛠️ Low Priority & Architectural Risk

### 7. "Unknown Party" Written as Literal Value
* **Issue**: When a politician's party cannot be determined, the pipeline writes "Unknown Party" as a literal string. This pollutes frontend filters as if it were a real party name.
* **Suggested Fix**:
  - Keep the party field empty/blank or set to `null` internally.
  - Display it as "party unconfirmed" on the UI, and programmatically resolve the party from `entities.json` by matching the politician's name where possible.

### 8. Single Point of Failure: `entities.json`
* **Issue**: If a foreign leader or junk entity is accidentally committed to `entities.json` in the `satya-entity-library` repo, it will bypass all filters across all components, including the promise tracker.
* **Suggested Fix**:
  - Place a strict nationality, validity, and manual-gate filter at the entity library's auto-add step (the point of entry), rather than relying only on consumer-side filters.
