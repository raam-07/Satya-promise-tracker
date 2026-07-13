import json
import collections

with open("/Users/mac/Downloads/Code/Satya/Satya-promise-tracker/promises.json", "r") as f:
    data = json.load(f)

promises = data.get("promises", [])
print(f"Total promises: {len(promises)}")

categories = [p.get("category") for p in promises]
category_counts = collections.Counter(categories)

print("\nCurrent Category Distribution:")
for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat}: {count}")

canonical_categories = [
    "jobs/employment", "economy", "farmers/agriculture", "health", 
    "education", "infrastructure", "welfare", "corruption/governance", 
    "law_and_order", "other"
]

invalid_promises = []
for p in promises:
    cat = p.get("category")
    if cat not in canonical_categories:
        invalid_promises.append((p.get("id"), p.get("person"), p.get("category"), p.get("promise")))

print(f"\nFound {len(invalid_promises)} promises with invalid/non-canonical categories:")
for pid, person, cat, text in invalid_promises:
    print(f"  ID: {pid} | Politician: {person} | Category: {cat} | Promise: {text[:60]}")
