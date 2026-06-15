import os
import sys
import json
# Add parent path to import pipeline
sys.path.append(os.path.dirname(__file__))
from promise_tracker_pipeline import (
    regex_pre_screen,
    run_stage1_noise_filter,
    run_stage2_extractor,
    run_stage3_critic,
    init_llm,
    MODEL_2B_PATH,
    MODEL_9B_PATH
)

# Mock Articles for Testing
mock_articles = [
    {
        "id": 1,
        "title": "India beats Australia by 6 wickets in T20 series",
        "content": "In a spectacular performance in 2026, the Indian cricket team beat Australia by 6 wickets in Mumbai.",
        "expected_regex": False,
        "expected_stage1": None,
        "description": "Sports article (should fail regex pre-screen)"
    },
    {
        "id": 2,
        "title": "Debate in assembly over historical welfare allocations",
        "content": "Politicians debated yesterday about welfare allocations and old schemes. The budget targets of 2022 were discussed heavily.",
        "expected_regex": True,
        "expected_stage1": False,
        "description": "General political noise/debate (should pass regex but fail Gemma 2B noise filter)"
    },
    {
        "id": 3,
        "title": "Government pledges to launch new electric vehicle subsidy scheme by 2028",
        "content": "Addressing the industrial summit, Chief Minister Patel declared: 'We guarantee that our government will launch a new electric vehicle subsidy scheme by 2028 to transition 50% of public transit to clean energy.'",
        "expected_regex": True,
        "expected_stage1": True,
        "description": "Actual concrete promise (should pass regex, Gemma 2B, and Gemma 9B)"
    }
]

def download_models_locally():
    from huggingface_hub import hf_hub_download
    print("Checking for local models...")
    os.makedirs("./models", exist_ok=True)
    
    if not os.path.exists(MODEL_2B_PATH):
        print(f"Downloading Gemma 2B model to: {MODEL_2B_PATH}...")
        hf_hub_download(repo_id="bartowski/gemma-2-2b-it-GGUF", filename="gemma-2-2b-it-Q4_K_M.gguf", local_dir="./models")
    else:
        print("Gemma 2B model already present.")
        
    if not os.path.exists(MODEL_9B_PATH):
        print(f"Downloading Gemma 9B model to: {MODEL_9B_PATH}...")
        hf_hub_download(repo_id="bartowski/gemma-2-9b-it-GGUF", filename="gemma-2-9b-it-Q4_K_M.gguf", local_dir="./models")
    else:
        print("Gemma 9B model already present.")

def run_tests():
    print("==================================================")
    print("           PIPELINE DRY RUN TEST SUITE             ")
    print("==================================================")

    # 1. Test Regex Pre-Screen (Fast)
    print("\n--- Testing Stage 0: Regex Pre-Screen ---")
    regex_passed = []
    for art in mock_articles:
        result = regex_pre_screen(art["title"], art["content"])
        passed = (result == art["expected_regex"])
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {art['description']}: Got Regex={result} (Expected={art['expected_regex']})")
        if result:
            regex_passed.append(art)

    # 2. Check if user wants to run LLM tests locally (requires downloading 7GB models)
    if not os.path.exists(MODEL_2B_PATH) or not os.path.exists(MODEL_9B_PATH):
        print("\n[NOTE] Local Gemma models not found. Skipped local LLM inference tests.")
        print("If you want to run the full LLM test suite locally, run:")
        print("  python3 test_pipeline.py --download-models")
        return

    print("\nInitializing local Gemma engines (this may take a few seconds)...")
    llm_2b = init_llm(MODEL_2B_PATH, 2048)
    llm_9b = init_llm(MODEL_9B_PATH, 4096)

    if not llm_2b or not llm_9b:
        print("[ERROR] Failed to load local Gemma engines. Exiting tests.")
        return

    # 3. Test LLM Stages
    print("\n--- Testing Stage 1 & 2 & 3: LLM Inference ---")
    for art, content in [(a, a["content"]) for a in regex_passed]:
        print(f"\nEvaluating: {art['title']}")
        
        # Stage 1: Noise Filter
        stage1_res = run_stage1_noise_filter(llm_2b, art["title"], content)
        passed_stage1 = (stage1_res == art["expected_stage1"])
        status_s1 = "PASSED" if passed_stage1 else "FAILED"
        print(f"  - Stage 1 Noise Filter: {stage1_res} ({status_s1})")
        
        if not stage1_res:
            continue
            
        # Stage 2: Extractor
        print("  - Running Stage 2 Extractor...")
        extracted_json = run_stage2_extractor(llm_9b, art["title"], content, [])
        if not extracted_json:
            print("  - [FAILED] Stage 2: Failed to extract valid JSON payload")
            continue
        print(f"  - Stage 2 JSON Extracted: {json.dumps(extracted_json)}")
        
        # Stage 3: Critic
        print("  - Running Stage 3 Adversarial Critic...")
        approved, msg = run_stage3_critic(llm_9b, content, extracted_json)
        print(f"  - Stage 3 Critic: Approved={approved} ({msg})")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download-models":
        download_models_locally()
    run_tests()
