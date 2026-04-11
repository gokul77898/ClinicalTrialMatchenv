"""
Realistic Cases — loader, evaluator, and scorer.

Pure Python, no heavy dependencies. Evaluates patient-trial eligibility
with full inclusion/exclusion trace and tie-breaking scoring.
"""

import json
import os
from typing import Any

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "realistic_cases.json",
)

_cache: list | None = None


def _load_all() -> list:
    global _cache
    if _cache is None:
        with open(_DATA_PATH, "r") as f:
            _cache = json.load(f)
    return _cache


def list_cases() -> list[dict]:
    """Return list of {case_id, label} for dropdown."""
    return [{"case_id": c["case_id"], "label": c["label"]} for c in _load_all()]


def load_case(case_id: str) -> dict | None:
    """Load a single case by ID."""
    for c in _load_all():
        if c["case_id"] == case_id:
            return c
    return None


# -----------------------------------------------
# FIELD RESOLVER
# -----------------------------------------------

def _resolve_field(patient: dict, field: str) -> Any:
    """Resolve a dotted field path like 'lab_values.hb' from patient dict."""
    parts = field.split(".")
    val = patient
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


# -----------------------------------------------
# CRITERIA EVALUATOR
# -----------------------------------------------

def _eval_criterion(patient: dict, crit: dict) -> tuple[bool, str]:
    """Evaluate one criterion. Returns (pass, reason)."""
    field = crit["field"]
    op = crit["op"]
    expected = crit["value"]
    actual = _resolve_field(patient, field)

    if actual is None:
        return False, f"{field} is unknown"

    # contains (for comorbidities list)
    if op == "contains":
        if isinstance(actual, list):
            passed = expected in actual
        else:
            passed = str(expected) in str(actual)
        verb = "contains" if passed else "does not contain"
        return passed, f"{field} {verb} '{expected}'"

    # comparison ops
    ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">=": lambda a, b: float(a) >= float(b),
        "<=": lambda a, b: float(a) <= float(b),
        ">":  lambda a, b: float(a) > float(b),
        "<":  lambda a, b: float(a) < float(b),
    }

    try:
        passed = ops[op](actual, expected)
    except (ValueError, TypeError, KeyError):
        return False, f"{field}: cannot compare {actual} {op} {expected}"

    symbol = "✔" if passed else "✖"
    return passed, f"{symbol} {field}: {actual} {op} {expected}"


def evaluate_trial(patient: dict, trial: dict) -> dict:
    """Evaluate one trial against a patient.

    Returns:
        {
            "trial_id": str,
            "inclusion_pass": bool,
            "exclusion_triggered": bool,
            "score": float,
            "verdict": str,
            "inclusion_reasons": [str],
            "exclusion_reasons": [str],
        }
    """
    inc_pass = True
    inc_reasons = []
    for crit in trial.get("inclusion", []):
        ok, reason = _eval_criterion(patient, crit)
        inc_reasons.append(reason)
        if not ok:
            inc_pass = False

    exc_triggered = False
    exc_reasons = []
    for crit in trial.get("exclusion", []):
        ok, reason = _eval_criterion(patient, crit)
        if ok:  # exclusion matched → patient is excluded
            exc_triggered = True
            exc_reasons.append(f"EXCLUDED: {reason}")
        else:
            exc_reasons.append(f"Clear: {reason}")

    # Scoring (only if eligible)
    score = 0.0
    if inc_pass and not exc_triggered:
        score += 2.0
        meta = trial.get("meta", {})
        score += meta.get("quality", 0.5) * 0.3
        score += (1.0 / max(1, meta.get("deadline_days", 90))) * 0.2
        score += meta.get("capacity_ratio", 0.5) * 0.2

    # Verdict
    if not inc_pass:
        verdict = "REJECTED — failed inclusion criteria"
    elif exc_triggered:
        verdict = "REJECTED — exclusion triggered"
    else:
        verdict = f"ELIGIBLE — score {score:.3f}"

    return {
        "trial_id": trial["trial_id"],
        "cancer_type": trial.get("cancer_type", "?"),
        "inclusion_pass": inc_pass,
        "exclusion_triggered": exc_triggered,
        "score": score,
        "verdict": verdict,
        "inclusion_reasons": inc_reasons,
        "exclusion_reasons": exc_reasons,
        "meta": trial.get("meta", {}),
    }


def evaluate_case(case: dict) -> dict:
    """Evaluate all trials for a case. Returns full decision trace."""
    patient = case["patient"]
    results = []
    for trial in case["trials"]:
        results.append(evaluate_trial(patient, trial))

    # Pick best
    eligible = [r for r in results if r["inclusion_pass"] and not r["exclusion_triggered"]]
    eligible.sort(key=lambda r: r["score"], reverse=True)

    selected = eligible[0]["trial_id"] if eligible else None
    correct = case.get("correct_trial")
    success = (selected == correct) if correct else (selected is None)

    return {
        "case_id": case["case_id"],
        "label": case["label"],
        "patient": patient,
        "trial_results": results,
        "eligible": [r["trial_id"] for r in eligible],
        "selected": selected,
        "correct_trial": correct,
        "success": success,
    }


# -----------------------------------------------
# FORMATTING (for UI)
# -----------------------------------------------

def fmt_patient(patient: dict) -> str:
    bio = patient.get("biomarkers", {})
    labs = patient.get("lab_values", {})
    lines = [
        f"Age:          {patient.get('age', '?')}",
        f"Gender:       {patient.get('gender', '?')}",
        f"Cancer Type:  {patient.get('cancer_type', '?')}",
        f"Stage:        {patient.get('stage', '?')}",
        "",
        "Biomarkers:",
        f"  EGFR:   {bio.get('EGFR', '?')}",
        f"  ALK:    {bio.get('ALK', '?')}",
        f"  PD_L1:  {bio.get('PD_L1', '?')}",
        "",
        "Lab Values:",
        f"  HB:         {labs.get('hb', 'unknown')}",
        f"  WBC:        {labs.get('wbc', 'unknown')}",
        f"  Creatinine: {labs.get('creatinine', 'unknown')}",
    ]
    comorbs = patient.get("comorbidities", [])
    if comorbs:
        lines += ["", "Comorbidities:"]
        for c in comorbs:
            lines.append(f"  - {c}")
    return "\n".join(lines)


def fmt_trials_evaluated(trial_results: list) -> str:
    lines = []
    for r in trial_results:
        tag = "✔ ELIGIBLE" if (r["inclusion_pass"] and not r["exclusion_triggered"]) else "✖ REJECTED"
        lines.append(f"{'─' * 40}")
        lines.append(f"{r['trial_id']}  [{r['cancer_type']}]  {tag}")
        if r["inclusion_pass"] and not r["exclusion_triggered"]:
            lines.append(f"  Score: {r['score']:.3f}")
        meta = r.get("meta", {})
        if meta:
            lines.append(f"  Quality: {meta.get('quality','?')}  |  "
                         f"Deadline: {meta.get('deadline_days','?')}d  |  "
                         f"Capacity: {meta.get('capacity_ratio','?')}")
        lines.append("")
        lines.append("  Inclusion:")
        for reason in r["inclusion_reasons"]:
            lines.append(f"    {reason}")
        if r["exclusion_reasons"]:
            lines.append("  Exclusion:")
            for reason in r["exclusion_reasons"]:
                lines.append(f"    {reason}")
        lines.append("")
    return "\n".join(lines).strip()


def fmt_decision(result: dict) -> str:
    lines = []

    # Why selected
    sel = result["selected"]
    if sel:
        lines.append(f"SELECTED: {sel}")
        eligible = result["eligible"]
        if len(eligible) > 1:
            lines.append(f"  ({len(eligible)} eligible — picked highest score)")
        lines.append("")
    else:
        lines.append("NO ELIGIBLE TRIAL FOUND")
        lines.append("")

    # Why others rejected
    for tr in result["trial_results"]:
        if tr["trial_id"] == sel:
            continue
        if not tr["inclusion_pass"]:
            failed = [r for r in tr["inclusion_reasons"] if r.startswith("✖")]
            reason = failed[0] if failed else "failed inclusion"
            lines.append(f"  ✖ {tr['trial_id']}: {reason}")
        elif tr["exclusion_triggered"]:
            exc = [r for r in tr["exclusion_reasons"] if r.startswith("EXCLUDED")]
            reason = exc[0] if exc else "exclusion triggered"
            lines.append(f"  ✖ {tr['trial_id']}: {reason}")
        elif tr["trial_id"] != sel:
            lines.append(f"  ○ {tr['trial_id']}: eligible but lower score ({tr['score']:.3f})")

    lines.append("")
    correct = result.get("correct_trial")
    if correct:
        match = "✅ CORRECT" if result["success"] else f"❌ WRONG (expected {correct})"
    else:
        match = "✅ CORRECT (no trial eligible)" if result["success"] else "❌ WRONG"
    lines.append(f"Result: {match}")
    return "\n".join(lines)
