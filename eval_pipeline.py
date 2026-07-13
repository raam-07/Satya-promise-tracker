import os
import sys
import json
import logging
import time

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure we can import from the local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from promise_tracker_pipeline import (
    run_stage2_extractor,
    run_stage3_critic,
    init_llm,
    is_valid_indian_politician,
    is_known_politician,
    quote_is_verbatim_in_source,
    is_atomic_claim,
    MODEL_9B_PATH,
    ask_llm_if_same_promise
)

# Mock politicians registry for test cases
mock_known_politicians = {
    "siddaramaiah",
    "mallikarjun kharge",
    "basangouda patil yatnal"
}

# Mock existing promises registry to test duplicate detection and progress mapping
mock_existing_promises = [
    {
        "id": "p001",
        "person": "Siddaramaiah",
        "party": "Congress",
        "role": "Chief Minister",
        "promise": "We are waiving crop loans up to Rs 1 lakh to support our farmers facing drought.",
        "status": "ongoing",
        "category": "farmers/agriculture"
    }
]

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_articles.json")
    if not os.path.exists(dataset_path):
        logging.critical(f"Dataset not found at {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        cases = json.load(f)
        
    logging.info(f"Loaded {len(cases)} evaluation test cases.")
    
    # Initialize the Qwen 14B model
    if not os.path.exists(MODEL_9B_PATH):
        logging.info("Model file not found. Downloading Qwen 14B GGUF model via huggingface_hub...")
        os.makedirs(os.path.dirname(MODEL_9B_PATH), exist_ok=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="bartowski/Qwen2.5-14B-Instruct-GGUF",
            filename="Qwen2.5-14B-Instruct-Q5_K_M.gguf",
            local_dir=os.path.dirname(MODEL_9B_PATH),
            local_dir_use_symlinks=False
        )
        
    logging.info(f"Loading Qwen 14B validation engine from {MODEL_9B_PATH}...")
    llm = init_llm(MODEL_9B_PATH, 8192)
    if not llm:
        logging.critical("Could not load LLM engine. Please check system RAM and llama-cpp-python compilation.")
        sys.exit(1)
        
    results = []
    
    for case in cases:
        case_id = case["id"]
        scenario = case["scenario"]
        title = case["title"]
        content = case["content"]
        expected = case["expected"]
        
        logging.info(f"\n==================================================")
        logging.info(f"TEST CASE #{case_id} [{scenario.upper()}]: {title[:50]}...")
        logging.info(f"==================================================")
        
        # Step 1: Run Stage 2 Extractor on RAW content
        logging.info("Executing Stage 2 Extractor...")
        extracted_json = run_stage2_extractor(llm, title, content, mock_existing_promises)
        
        should_extract = expected["should_extract"]
        
        if not extracted_json or extracted_json == {}:
            # Handle empty extract
            if not should_extract:
                logging.info(f"PASS: Correctly rejected/skipped case.")
                results.append({"id": case_id, "scenario": scenario, "passed": True, "details": "Correctly skipped non-promise"})
            else:
                logging.warning(f"FAIL: Failed to extract promise which was expected.")
                results.append({"id": case_id, "scenario": scenario, "passed": False, "details": "Failed to extract valid promise"})
            continue
            
        logging.info(f"Extracted JSON:\n{json.dumps(extracted_json, indent=2)}")
        
        # Step 2: Verification Guardrails
        politician_name = extracted_json.get("politician", "")
        supporting_quote = extracted_json.get("supporting_quote", "")
        promise_text = extracted_json.get("promise_text", "")
        evidence_type = extracted_json.get("evidence_type", "declaration")
        
        valid_politician = is_valid_indian_politician(politician_name) and is_known_politician(politician_name, mock_known_politicians)
        verbatim_quote = quote_is_verbatim_in_source(supporting_quote, content)
        atomic = is_atomic_claim(promise_text)
        
        logging.info(f"Guardrail Checks: Valid Politician={valid_politician}, Verbatim Quote={verbatim_quote}, Atomic={atomic}")
        
        # Step 3: Stage 3 Critic Check
        logging.info("Executing Stage 3 Critic...")
        approved, critic_msg = run_stage3_critic(llm, content, extracted_json)
        logging.info(f"Critic Approved={approved}, Message={critic_msg}")
        
        # Auto-merge / Duplicate check simulation
        is_new = extracted_json.get("is_new_promise", True)
        matched_id = extracted_json.get("matched_existing_promise_id")
        
        # Simple duplicate matching check: if duplicate/progress scenario, verify same promise check
        if scenario in ["progress_update", "duplicate"]:
            # Test duplicate adjudication
            is_same = ask_llm_if_same_promise(llm, promise_text, mock_existing_promises[0]["promise"])
            logging.info(f"LLM Same Promise Adjudication: {is_same}")
            if is_same:
                is_new = False
                matched_id = "p001"
        
        # Evaluate decision correctness
        passed = False
        details = ""
        
        if not should_extract:
            # We expected it to fail or be rejected by checks/critic
            if not approved or not verbatim_quote or not valid_politician:
                passed = True
                details = f"Correctly rejected. (Verbatim={verbatim_quote}, Valid Pol={valid_politician}, Critic={approved})"
            else:
                details = "FAIL: Extracted and approved a claim that should have been rejected."
        else:
            # We expected successful extraction and validation
            if approved and verbatim_quote and valid_politician:
                # Check politician match
                pol_ok = expected["politician"].lower() in politician_name.lower() or politician_name.lower() in expected["politician"].lower()
                evidence_ok = (expected["evidence_type"] == evidence_type)
                new_ok = (expected["is_new"] == is_new)
                
                if pol_ok and evidence_ok and new_ok:
                    passed = True
                    details = f"PASS: Successfully extracted, validated, and matched. matched_id={matched_id}"
                else:
                    details = f"FAIL: Mismatch in fields. Expected (pol={expected['politician']}, type={expected['evidence_type']}, is_new={expected['is_new']}), Got (pol={politician_name}, type={evidence_type}, is_new={is_new})"
            else:
                details = f"FAIL: Guardrail or Critic rejected valid promise. (Verbatim={verbatim_quote}, Valid Pol={valid_politician}, Critic={approved})"
                
        if passed:
            logging.info(f"SUCCESS: {details}")
        else:
            logging.error(f"FAILURE: {details}")
            
        results.append({
            "id": case_id,
            "scenario": scenario,
            "passed": passed,
            "details": details
        })
        
    # Output evaluation summary
    print("\n" + "="*50)
    print("EVALUATION RUN REPORT SUMMARY")
    print("="*50)
    passed_cnt = sum(1 for r in results if r["passed"])
    total_cnt = len(results)
    accuracy = (passed_cnt / total_cnt) * 100
    
    print(f"Total Test Cases: {total_cnt}")
    print(f"Passed Cases:    {passed_cnt}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print("-"*50)
    
    print("| ID | Scenario | Status | Details |")
    print("|---|---|---|---|")
    for r in results:
        status_str = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"| {r['id']} | {r['scenario']} | {status_str} | {r['details']} |")
    print("="*50)

if __name__ == "__main__":
    main()
