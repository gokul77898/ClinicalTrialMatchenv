"""
ClinicalTrialMatchEnv - Baseline Inference Script

Runs an LLM agent against all 3 clinical trial matching tasks.

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your-api-key-here"
    python inference.py

Environment Variables:
    API_BASE_URL: Base URL for OpenAI-compatible API
    MODEL_NAME:   Model identifier to use
    HF_TOKEN:     API key for authentication
"""

import os
import sys
import json
import time
import subprocess
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct:novita")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

ENV_BASE_URL = "http://localhost:7860"
MAX_STEPS_PER_TASK = 18
TASKS = [
    "single_match",
    "hidden_exclusion",
    "ambiguous_match",
    "competing_trials",
    "contradictory_info",
    "multi_patient",
    "logical_inference"
]

# -----------------------------------------------
# SERVER MANAGEMENT
# -----------------------------------------------

def start_server():
    """Start the FastAPI environment server."""
    print("Starting environment server...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "api.server:app",
         "--host", "0.0.0.0",
         "--port", "7860",
         "--log-level", "error"],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    # Wait for server to be ready (max 30 seconds)
    for i in range(30):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i+1} seconds")
                return proc
        except Exception:
            pass
        time.sleep(1)

    proc.terminate()
    raise RuntimeError("Server failed to start within 30 seconds")

def stop_server(proc):
    """Stop the FastAPI environment server."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("Server stopped")

# -----------------------------------------------
# ENVIRONMENT INTERACTION
# -----------------------------------------------

def env_reset(task_id: str) -> dict:
    """Reset environment for a specific task."""
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=10
    )
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    """Take a step in the environment."""
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=10
    )
    r.raise_for_status()
    return r.json()

def env_get_tasks() -> list:
    """Get list of available tasks."""
    r = requests.get(f"{ENV_BASE_URL}/tasks", timeout=10)
    r.raise_for_status()
    return r.json()["tasks"]

# -----------------------------------------------
# LLM AGENT
# -----------------------------------------------

SYSTEM_PROMPT = """You are a clinical trial matching agent.

Your job: Match a cancer patient to the correct clinical trial.

RULES:
- You must respond with ONLY a JSON object
- No explanation, no markdown, no extra text
- Just the raw JSON action

AVAILABLE ACTIONS:

Check a trial's eligibility criteria:
{"type": "check_criteria", "trial_id": "TRIAL-XXX-1234"}

Select the trial you think matches:
{"type": "select_trial", "trial_id": "TRIAL-XXX-1234"}

Flag a contradiction in the patient chart:
{"type": "flag_contradiction", "reason": "lab values inconsistent with stage"}

Investigate a conflicting field value:
{"type": "investigate_conflict", "field": "stage"}

Switch to a different patient case (multi-patient mode):
{"type": "switch_case", "case_id": "case_2"}

End the episode (REQUIRED to finish):
{"type": "resolve"}

When you call check_criteria, you receive different
levels of detail based on task difficulty:

EASY tasks: Full details including which rules passed/failed
and a plain-English summary. Use this to select directly.

MEDIUM tasks: Boolean flags only + a hint. You must
investigate patient fields to understand WHY a check failed.

HARD/EXPERT tasks: Boolean flags only. No hints.
You must reason from scratch using patient field values.

Always investigate patient fields BEFORE deciding on hard tasks.

Patient data includes:
- biomarkers.EGFR_expression / ALK_expression (0.0-1.0 confidence)
- lab_value_trends: direction (increasing/decreasing/stable) and rate
- comorbidities with severity levels (mild/moderate/severe)
- prior_treatments list

Trial metadata:
- max_patients / enrolled_patients / has_capacity: full trials are ineligible
- days_until_deadline / is_urgent: urgent trials (<=14 days) need priority
- trial_score: quality score (0.0-1.0) for comparing eligible trials
- has_biomarker_requirements: if true, trial requires specific biomarkers
- has_expression_thresholds: if true, you MUST investigate biomarkers.EGFR_expression
  and biomarkers.ALK_expression before selecting that trial. The threshold may be
  borderline — patient value could be just above or just below the minimum required.
- has_interaction_rules: if true, be careful — this trial may exclude patients based on
  COMBINATIONS of fields (e.g. age > 65 AND creatinine > 1.4). Investigate multiple
  fields before deciding.

Additional actions:
- switch_case: switch active patient in multi-patient mode (requires case_id)
- flag_contradiction: flag contradictory patient data (requires reason string)
- investigate_conflict: resolve conflicting field values (requires field name)

IMPORTANT — DATA QUALITY ISSUES:
- Some patient lab values may be 'unknown' (lab test not performed).
  If a trial requires a lab value that is unknown, the patient CANNOT
  be matched to that trial safely. Always investigate lab values first.
- Some patients have conflicting field values (e.g. stage reported as III
  but notes say II). If you see 'conflicting report' when investigating,
  use investigate_conflict action to get the full details.
- Patient data_freshness_days shows how old the data is. Stale data
  (>90 days) may not reflect current patient state.

STRATEGY:
1. First investigate key patient fields (cancer_type, stage, biomarkers, lab_values)
2. Check criteria for each trial
3. If a trial fails, investigate WHY by checking patient fields
4. If you see 'unknown' or 'conflicting report', handle it before deciding
5. Select the trial where ALL checks pass
6. Call resolve to finish

DO NOT guess. Investigate before deciding.

CRITICAL: You MUST call resolve within 20 steps or you score 0.
CRITICAL: Respond with JSON only. No other text."""

def build_user_message(observation: dict, step_history: list) -> str:
    """Build the user message for the LLM."""
    patient = observation["patient"]
    trials = observation["available_trials"]
    steps_left = observation["max_steps"] - observation["steps_taken"]
    checked = observation["checked_trials"]
    selected = observation["selected_trial_id"]

    # Build trial list
    trial_lines = []
    for t in trials:
        status = "CHECKED" if t["trial_id"] in checked else "NOT CHECKED"
        cap = f"cap={t.get('enrolled_patients',0)}/{t.get('max_patients',10)}" if 'max_patients' in t else ""
        deadline = f"deadline={t.get('days_until_deadline','?')}d" if 'days_until_deadline' in t else ""
        score = f"score={t.get('trial_score','?')}" if 'trial_score' in t else ""
        extras = " | ".join(x for x in [cap, deadline, score] if x)
        trial_lines.append(
            f"  {t['trial_id']} | {t['cancer_type']} | {status} | {extras}"
        )
    trials_str = "\n".join(trial_lines)

    # Build recent history (last 5 only)
    recent = step_history[-5:] if len(step_history) > 5 else step_history
    history_lines = []
    for h in recent:
        action = h["action"]
        reward = h["reward"]
        result = h.get("result_info", "")
        summary = h.get("eligibility_summary", "")
        line = f"  {action} -> reward={reward:.2f} {result}"
        if summary:
            line += f" | {summary}"
        history_lines.append(line)
    history_str = "\n".join(history_lines) if history_lines else "  (none)"

    # Urgency message based on steps left
    if steps_left <= 3:
        urgency = f"URGENT: Only {steps_left} steps left! You MUST select_trial then resolve NOW!"
    elif steps_left <= 6:
        urgency = f"WARNING: Only {steps_left} steps left. Wrap up soon."
    else:
        urgency = f"Steps remaining: {steps_left}"

    msg = f"""PATIENT: {patient['age']}yo {patient['gender']}, {patient['cancer_type']}, stage {patient['stage']}
Biomarkers: EGFR={patient['biomarkers']['EGFR']}, ALK={patient['biomarkers']['ALK']}, PD_L1={patient['biomarkers']['PD_L1']}
Labs: HB={patient['lab_values']['hb'] if patient['lab_values']['hb'] is not None else 'unknown'}, WBC={patient['lab_values']['wbc'] if patient['lab_values']['wbc'] is not None else 'unknown'}, Creatinine={patient['lab_values']['creatinine'] if patient['lab_values']['creatinine'] is not None else 'unknown'}
Comorbidities: {patient.get('comorbidities', [])}

TRIALS:
{trials_str}

SELECTED: {selected if selected else 'NONE - you must select then resolve'}
{urgency}

RECENT ACTIONS:
{history_str}

Respond with ONE JSON action only:"""

    return msg

def get_llm_action(
    client: OpenAI,
    observation: dict,
    step_history: list,
    task_max_steps: int = 18,
    retry: int = 3
) -> dict:
    """Get next action from LLM with smart fallback."""

    steps_taken = observation["steps_taken"]
    max_steps = observation["max_steps"]
    steps_left = max_steps - steps_taken
    selected = observation["selected_trial_id"]
    checked = observation["checked_trials"]
    trials = [t["trial_id"] for t in observation["available_trials"]]

    # HARD RULE 1: If only 2 steps left, force select+resolve
    if steps_left <= 2:
        if selected is None:
            # Pick most promising trial
            unchecked = [t for t in trials if t not in checked]
            target = unchecked[0] if unchecked else trials[0]
            print(f"  [FORCED] Selecting trial (steps critical): {target}")
            return {"type": "select_trial", "trial_id": target}
        else:
            print(f"  [FORCED] Resolving (steps critical)")
            return {"type": "resolve"}

    # HARD RULE 2: If selected and last 2 actions were not resolve, force resolve
    if selected is not None:
        recent_types = [h["action"].get("type") for h in step_history[-2:]]
        if "resolve" not in recent_types and steps_left <= 4:
            print(f"  [FORCED] Resolving (trial selected, low steps)")
            return {"type": "resolve"}

    # HARD RULE 3: If all trials checked and none selected, pick the best one
    all_checked = all(t in checked for t in trials)
    if all_checked and selected is None:
        print(f"  [FORCED] All trials checked, selecting first: {trials[0]}")
        return {"type": "select_trial", "trial_id": trials[0]}

    user_msg = build_user_message(observation, step_history)

    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0,
                max_tokens=80
            )

            raw = response.choices[0].message.content.strip()
            print(f"  [LLM raw]: {raw[:150]}")

            # Aggressive JSON extraction
            raw_clean = raw

            # Remove markdown
            if "```" in raw_clean:
                parts = raw_clean.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw_clean = part
                        break

            # Find JSON object in response
            start = raw_clean.find("{")
            end = raw_clean.rfind("}") + 1
            if start >= 0 and end > start:
                raw_clean = raw_clean[start:end]

            action = json.loads(raw_clean)

            # Validate action has required type
            if "type" not in action:
                raise ValueError("Action missing 'type' field")

            valid_types = {
                "investigate", "check_criteria",
                "select_trial", "resolve",
                "flag_contradiction", "switch_case",
                "investigate_conflict"
            }
            if action["type"] not in valid_types:
                raise ValueError(
                    f"Invalid action type: {action['type']}"
                )

            # Validate required fields
            if action["type"] == "investigate" and "field" not in action:
                action["field"] = "age"
            if action["type"] == "flag_contradiction" and "reason" not in action:
                action["reason"] = "chart inconsistency detected"
            if action["type"] == "investigate_conflict" and "field" not in action:
                action["field"] = "stage"
            if action["type"] == "switch_case" and "case_id" not in action:
                action["case_id"] = "case_1"
            if action["type"] in ("check_criteria", "select_trial"):
                if "trial_id" not in action:
                    unchecked = [t for t in trials if t not in checked]
                    action["trial_id"] = (
                        unchecked[0] if unchecked else trials[0]
                    )

            return action

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  [Attempt {attempt+1}] Parse error: {e}")
            if attempt == retry - 1:
                # Smart fallback based on state
                if selected is not None:
                    print("  [Fallback] Resolving (trial already selected)")
                    return {"type": "resolve"}
                unchecked = [t for t in trials if t not in checked]
                if unchecked:
                    print(f"  [Fallback] Checking unchecked trial: {unchecked[0]}")
                    return {
                        "type": "check_criteria",
                        "trial_id": unchecked[0]
                    }
                print(f"  [Fallback] Selecting: {trials[0]}")
                return {"type": "select_trial", "trial_id": trials[0]}

        except Exception as e:
            print(f"  [Attempt {attempt+1}] API error: {e}")
            if attempt == retry - 1:
                if selected is not None:
                    return {"type": "resolve"}
                return {
                    "type": "check_criteria",
                    "trial_id": trials[0]
                }
            time.sleep(2)

    return {"type": "resolve"}

# -----------------------------------------------
# RUN ONE TASK
# -----------------------------------------------

def get_max_steps(task_id: str) -> int:
    """Get maximum steps allowed for a task."""
    if task_id == "multi_patient":
        return 25  # needs more steps for 3 patients
    return 18

def _log(tag: str, data: dict):
    """Emit a structured log line: [TAG] {json}"""
    print(f"[{tag}] {json.dumps(data)}", flush=True)


def run_task(client: OpenAI, task_id: str) -> dict:
    """Run LLM agent on one task."""
    from datetime import datetime, timezone

    observation = env_reset(task_id)

    task_max_steps = get_max_steps(task_id)

    _log("START", {
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient": {
            "age": observation["patient"]["age"],
            "gender": observation["patient"]["gender"],
            "cancer_type": observation["patient"]["cancer_type"],
            "stage": observation["patient"]["stage"],
        },
        "num_trials": len(observation["available_trials"]),
        "max_steps": task_max_steps,
    })

    step_history = []
    done = False
    grade = 0.0
    steps = 0
    final_info = {}

    while not done and steps < task_max_steps:
        steps += 1

        action = get_llm_action(client, observation, step_history, task_max_steps)

        try:
            result = env_step(action)

            observation = result["observation"]
            reward_val = result["reward"]["value"]
            done = result["done"]
            info = result["info"]
            reason = result["reward"].get("reason", "")

            _log("STEP", {
                "task_id": task_id,
                "step": steps,
                "action": action,
                "reward": round(reward_val, 4),
                "done": done,
                "info": reason,
            })

            # Capture eligibility summary if available
            eligibility_summary = info.get("summary", "")

            step_history.append({
                "action": action,
                "reward": reward_val,
                "done": done,
                "result_info": reason,
                "eligibility_summary": eligibility_summary
            })

            if done:
                grade = info.get("grade", 0.0)
                final_info = info
                break

        except requests.exceptions.HTTPError as e:
            _log("STEP", {
                "task_id": task_id,
                "step": steps,
                "action": action,
                "reward": 0.0,
                "done": False,
                "info": f"HTTP error: {e}",
            })
            if e.response.status_code == 400:
                break
            time.sleep(1)
        except Exception as e:
            _log("STEP", {
                "task_id": task_id,
                "step": steps,
                "action": action,
                "reward": 0.0,
                "done": False,
                "info": f"Error: {e}",
            })
            time.sleep(1)

    # If not done after loop, force resolve
    if not done:
        try:
            # First select a trial if none selected
            if observation["selected_trial_id"] is None:
                trials = [
                    t["trial_id"]
                    for t in observation["available_trials"]
                ]
                checked = observation["checked_trials"]
                target = checked[0] if checked else trials[0]
                env_step({
                    "type": "select_trial",
                    "trial_id": target
                })

            result = env_step({"type": "resolve"})
            grade = result["info"].get("grade", 0.0)
            final_info = result["info"]
            done = True
        except Exception as e:
            grade = 0.0

    correct = final_info.get("correct", False)

    _log("END", {
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "grade": round(grade, 4),
        "correct": correct,
        "steps": steps,
    })

    return {
        "task_id": task_id,
        "grade": grade,
        "correct": correct,
        "steps": steps,
        "done": done
    }

# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main():
    from datetime import datetime, timezone

    # Validate environment variables
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set")
        print("Usage: export HF_TOKEN=your-api-key")
        sys.exit(1)

    _log("START", {
        "run": "ClinicalTrialMatchEnv",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "tasks": TASKS,
    })

    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

    # Start environment server
    server_proc = None
    try:
        server_proc = start_server()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Run all tasks
    results = []
    try:
        for task_id in TASKS:
            try:
                result = run_task(client, task_id)
                results.append(result)
            except Exception as e:
                _log("END", {
                    "task_id": task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "grade": 0.0,
                    "correct": False,
                    "steps": 0,
                    "error": str(e),
                })
                results.append({
                    "task_id": task_id,
                    "grade": 0.0,
                    "correct": False,
                    "steps": 0,
                    "done": False,
                    "error": str(e)
                })
    finally:
        stop_server(server_proc)

    # Compute summary
    total_grade = 0.0
    for r in results:
        total_grade += r.get("grade", 0.0)
    avg_grade = total_grade / len(results) if results else 0.0

    _log("END", {
        "run": "ClinicalTrialMatchEnv",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "average_grade": round(avg_grade, 4),
        "results": [
            {
                "task_id": r["task_id"],
                "grade": round(r.get("grade", 0.0), 4),
                "correct": r.get("correct", False),
                "steps": r.get("steps", 0),
            }
            for r in results
        ],
    })

    # Save results to file
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "baseline_results.json"
    )
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "results": results,
            "average_grade": avg_grade
        }, f, indent=2)

    # Exit 0 if all tasks completed
    all_done = all(r.get("done", False) for r in results)
    sys.exit(0 if all_done else 1)

if __name__ == "__main__":
    main()
