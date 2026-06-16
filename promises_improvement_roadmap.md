# Satya Promise Tracker — Autonomous (Zero-Human) Ingestion Roadmap

This document defines the roadmap for achieving a **100% autonomous, zero-human** pipeline for tracking political promises. It frames all remaining gaps as mechanical, self-governing gates rather than tasks requiring human intervention.

---

## 🚨 High Priority (Trust, Duplication & Auto-Merging)

### 1. Uncertain Verdicts: Evidence-Threshold Auto-Resolver (Finding #1)
* **Gap**: Gated status suggestions are written to a review queue file (`review_promises.json`) which acts as a dead-end since there is no human operator in the loop.
* **Zero-Human Fix**: 
  - Treat the review file as a transient buffer or eliminate it in favor of an **evidence-threshold auto-resolver**.
  - Keep a promise at its current status (e.g. `ongoing`) and continuously collect evidence. 
  - A status is only allowed to flip to `kept` or `broken` once a strict threshold of **2 to 3 independent, validated media sources** agree on the new verdict. Until that threshold is met, the machine retains the current status automatically.

### 2. Borderline Duplicates: LLM Auto-DecDecider (Finding #2)
* **Gap**: Promises in the grey similarity zone (60% to 75%) are parked in the review queue, causing backlog stagnation.
* **Zero-Human Fix**:
  - Implement a secondary **yes/no LLM query** when a duplicate falls into the 60%–75% similarity zone.
  - Present the text of the existing promise and the new candidate to the model with a prompt: *"Are these two sentences referring to the exact same political promise/welfare goal? Reply YES or NO."*
  - The pipeline automatically merges the promise if `YES`, and creates a new one if `NO`, leaving no items parked.

### 3. New Promise Verdict Safeguard (Finding #3)
* **Gap**: A new promise could be erroneously created as "broken" off a single news article.
* **Zero-Human Fix**:
  - Hardcode all brand-new promises to be created with status `"ongoing"`.
  - A harsher status like `"broken"` or `"kept"` must be earned subsequently over time through the multi-source evidence resolver.

---

## 📈 Medium Priority (Auditability & Resiliency)

### 4. Append-Only Status Trajectory Logs (Finding #5)
* **Gap**: A promise's status is overwritten directly on the object. A prior status like "broken" can silently become "ongoing" with no historical log or audit trail.
* **Zero-Human Fix**:
  - Implement an append-only `status_history` log inside the promise schema:
    ```json
    "status_history": [
      {
        "status": "ongoing",
        "changed_at": "2026-06-15",
        "evidence_url": "https://..."
      },
      {
        "status": "broken",
        "changed_at": "2026-06-17",
        "evidence_url": "https://..."
      }
    ]
    ```
  - The pipeline appends to this log on every state change, preserving the complete trajectory of the verdict.

### 5. Loop Stagnation Watchdog & Signal (Finding #4)
* **Gap**: The self-loop stops on failure and remains halted until a human manually restarts it.
* **Zero-Human Fix**:
  - Implement a **watchdog check** on the row pointer.
  - If a write failure or stagnation is detected, auto-retry the database write 3 times.
  - If it remains stuck after 3 attempts, immediately abort the loop and trigger an **automated warning webhook/ping** or auto-create a GitHub Issue containing the diagnostic logs so the system self-signals the failure.

### 6. Accurate Promise Dates (`made_on`) (Finding #6)
* **Gap**: Old promises reported in news coverage today get tagged with today's scraping date as `made_on`.
* **Zero-Human Fix**:
  - Ask the Stage 2 Extractor to pull the promise declaration date from the text or quote.
  - If the model is unsure or no date is explicitly stated, default to the scrape date but label the metadata field as `"reported_on"` instead of `"made_on"`.

---

## 🛠️ Low Priority & Architectural Risk

### 7. Party Resolution & Null Gating (Finding #7)
* **Gap**: "Unknown Party" is written as a literal party string, polluting frontend filters.
* **Zero-Human Fix**:
  - If the party cannot be resolved from the article text or database fallback, cross-reference the politician name against the canonical `entities.json` registry to load their registered party.
  - If still unresolved, leave the field blank and show `"party unconfirmed"` on the UI, ensuring no fake party names enter the registry.

### 8. Category & Significance Taxonomy (Finding #8)
* **Gap**: Promises use the basic news-classifier taxonomy, lacking a citizen-facing classification or a promise weight tier.
* **Zero-Human Fix**:
  - Update the Stage 2 extraction prompt to yield two tags: the core topic and a significance weight tier (e.g. *Big Policy, Everyday Scheme, or Freebie/Slogan*).
  - The pipeline automatically processes both metrics to drive frontend filtering.

### 9. Entry-Gate Protection for `entities.json` (Finding #9)
* **Gap**: `entities.json` is the single point of failure; any garbage added there gets trusted downstream.
* **Zero-Human Fix**:
  - Implement a strict nationality, validity, and structural gate in the `satya-entity-library` updater.
  - Every candidate entity must pass a verification check at the *entry point* (where it is first added to the library) rather than relying on consumer repos to screen them.
