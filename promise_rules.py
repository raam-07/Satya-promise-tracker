"""Time rules for promises that were made without a deadline.

SatyaDheesh policy:
- A promise with no deadline is judged OPEN_ENDED_WINDOW_YEARS after it was
  made. If it is still "ongoing" then, it becomes "broken".
- An open-ended promise can be marked "kept" only after it has been monitored
  for KEPT_MIN_MONITOR_YEARS. Sustained outcomes ("end mafia raj", "free bus
  passes") can be reversed, so a short-lived result is not enough. A promise
  that has been delivered but not yet monitored long enough stays "ongoing"
  with a `monitoring` note; it is never broken by the 3-year rule, and it
  becomes "kept" once the monitoring period ends.

Promises with an explicit deadline are unaffected: they are judged at their
deadline and can be kept as soon as they are delivered.

The clock starts on `made_on`. For records no editor has verified, the
pipeline's `made_on` is sometimes an old date the model guessed, so the clock
starts at the later of `made_on` and `reported_on`. That can only delay a
verdict, never bring one forward by mistake.
"""
import datetime as dt
import re

OPEN_ENDED_WINDOW_YEARS = 3
KEPT_MIN_MONITOR_YEARS = 5

_NO_DEADLINE = {"", "ongoing", "none", "null", "n/a", "na", "no deadline"}


def parse_date(value):
    """'2019', '2019-04' or '2019-04-08' -> date (first day of the missing unit)."""
    m = re.match(r"^\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", str(value or ""))
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))
    except ValueError:
        return None


def add_years(day, years):
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 Feb
        return day.replace(year=day.year + years, day=28)


def has_explicit_deadline(promise):
    return str(promise.get("deadline") or "").strip().lower() not in _NO_DEADLINE


def clock_start(promise):
    made = parse_date(promise.get("made_on"))
    reported = parse_date(promise.get("reported_on"))
    if (promise.get("editorial") or {}).get("verified_on"):
        return made or reported
    dates = [d for d in (made, reported) if d]
    return max(dates) if dates else None


def implied_deadline(promise):
    """The date an open-ended promise is judged, or None if it has a real deadline."""
    if has_explicit_deadline(promise):
        return None
    start = clock_start(promise)
    return add_years(start, OPEN_ENDED_WINDOW_YEARS) if start else None


def kept_allowed(promise, today=None):
    """False while an open-ended promise is still inside its monitoring period."""
    if has_explicit_deadline(promise):
        return True
    start = clock_start(promise)
    if not start:
        return False
    return (today or dt.date.today()) >= add_years(start, KEPT_MIN_MONITOR_YEARS)


def monitoring_until(promise):
    """When a delivered open-ended promise can be marked kept."""
    start = clock_start(promise)
    return add_years(start, KEPT_MIN_MONITOR_YEARS) if start else None


def apply_time_rules(promises, today=None):
    """Write implied_deadline on open-ended promises, break the overdue ones and
    mark kept the delivered ones whose monitoring period has ended.

    Returns the ids whose status changed. Records an editor excluded are skipped.
    Editor-locked verdicts are included: these are editor rules, and they only
    move "ongoing", which a lock never set as a final decision.
    """
    today = today or dt.date.today()
    changed = []
    for p in promises:
        if (p.get("editorial") or {}).get("exclude"):
            continue
        due = implied_deadline(p)
        if due is None:
            p.pop("implied_deadline", None)
            continue
        p["implied_deadline"] = due.isoformat()
        if p.get("status") == "ongoing" and p.get("monitoring"):
            # Delivered: wait out the monitoring period, then mark kept.
            if kept_allowed(p, today):
                p["status"] = "kept"
                p.setdefault("status_history", []).append({
                    "status": "kept",
                    "changed_at": today.isoformat(),
                    "by": "rule",
                    "note": f"Delivered and monitored for {KEPT_MIN_MONITOR_YEARS} years.",
                })
                p["status_last_reviewed"] = today.isoformat()
                p.pop("monitoring", None)
                changed.append(p.get("id"))
            continue
        if p.get("status") == "ongoing" and today > due:
            p["status"] = "broken"
            p.setdefault("status_history", []).append({
                "status": "broken",
                "changed_at": today.isoformat(),
                "by": "rule",
                "note": f"No deadline was given; judged {OPEN_ENDED_WINDOW_YEARS} years after it was made ({due.isoformat()}).",
            })
            p["status_last_reviewed"] = today.isoformat()
            changed.append(p.get("id"))
    return changed
