"""Apply an editor-verified fix file to promises.json.

usage: python3 scripts/apply_editorial.py editorial/<date>.json [--dry-run]

Each change can set base fields ("set") and attach an "editorial" block.
The pipeline never writes the "editorial" field or the promise text, so
these survive later pipeline runs. The file is rewritten with the same
formatting the pipeline uses, so the diff stays small.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "promises.json"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if len(args) != 1:
        sys.exit(__doc__)

    fixes = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    raw = REGISTRY.read_text(encoding="utf-8")
    data = json.loads(raw)
    promises = data["promises"] if isinstance(data, dict) else data
    by_id = {str(p.get("id")): p for p in promises}

    # The fix file is named after the day the editor decided, e.g. editorial/2026-09-27.json.
    decided_on = Path(args[0]).stem[:10]

    for change in fixes["changes"]:
        pid = change["id"]
        p = by_id.get(pid)
        if p is None:
            print(f"{pid}: not found, skipped")
            continue
        for field, value in (change.get("set") or {}).items():
            print(f"{pid}: {field}: {p.get(field)!r} -> {value!r}")
            if field == "status" and value != p.get("status"):
                # Keep the public verdict history honest about who moved it and when.
                p.setdefault("status_history", []).append({
                    "status": value,
                    "changed_at": decided_on,
                    "by": "editor",
                    "note": change.get("why", ""),
                })
                p["status_last_reviewed"] = decided_on
            p[field] = value
        if "editorial" in change:
            ed = {k: v for k, v in change["editorial"].items() if v not in (None, [], "")}
            p["editorial"] = ed
            print(f"{pid}: editorial set ({', '.join(ed)})")

    if dry:
        print("dry run, nothing written")
        return
    # Same settings as save_promises() in the pipeline.
    REGISTRY.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REGISTRY.name}")


if __name__ == "__main__":
    main()
