"""
ClinicalTrialMatchEnv - OpenEnv-Compliant Inference Script

Runs a hybrid LLM+heuristic agent on a single clinical trial matching task.
LLM is used for initial reasoning (steps 1-2), then a fast deterministic
heuristic takes over to guarantee completion within budget.

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4.1-mini"
    export HF_TOKEN="your-api-key-here"
    python inference.py
"""

import os
import sys
import json
import time
import subprocess
import requests
from openai import OpenAI
from dotenv import load_dotenv
from src.graders import _clamp

load_dotenv()

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
ENV_NAME = "clinical_trial"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MAX_STEPS = 18
TASKS = ["single_match", "hidden_exclusion", "ambiguous_match"]

# -----------------------------------------------
# SERVER MANAGEMENT (auto-start if not running)
# -----------------------------------------------

_server_proc = None

def _ensure_server():
    """Start the env server if it isn't already running."""
    global _server_proc
    # Check if already up
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=2)
        if r.status_code == 200:
            return
    except Exception:
        pass

    # Auto-start
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    _server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "api.server:app", "--host", "0.0.0.0",
         "--port", "7860", "--log-level", "error"],
        env=env, cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(15):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Server failed to start")


def _stop_server():
    global _server_proc
    if _server_proc:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None

# -----------------------------------------------
# ENVIRONMENT INTERACTION
# -----------------------------------------------

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=10)
    r.raise_for_status()
    return r.json()

# -----------------------------------------------
# STRICT LOGGING (OpenEnv format only)
# -----------------------------------------------

def _fmt_action(action: dict) -> str:
    """Format action dict as compliant string: check_criteria("TRIAL-123")."""
    atype = action.get("type", "unknown")
    tid = action.get("trial_id")
    if atype in ("check_criteria", "select_trial") and tid:
        return f'{atype}("{tid}")'
    return f"{atype}()"


def _log_start(task: str):
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def _log_step(step: int, action_str: str, reward: float, done: bool, error: str = "null"):
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error}", flush=True)

def _log_end(success: bool, steps: int, rewards: list):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# -----------------------------------------------
# LLM CALL (used only for first 1-2 steps)
# -----------------------------------------------

SYSTEM_PROMPT = """You are a clinical trial matching agent. Respond with ONLY a JSON object.

ACTIONS:
{"type": "check_criteria", "trial_id": "TRIAL-XXX-1234"}
{"type": "select_trial", "trial_id": "TRIAL-XXX-1234"}
{"type": "resolve"}

STRATEGY: Check each trial's criteria. Select the one that passes inclusion
and does NOT trigger exclusion. Then resolve. JSON only, no explanation."""

def call_llm(client: OpenAI, observation: dict) -> dict | None:
    """Single LLM call with tight timeout. Returns parsed action or None."""
    patient = observation["patient"]
    trials = observation["available_trials"]
    checked = observation["checked_trials"]

    trial_lines = []
    for t in trials:
        status = "CHECKED" if t["trial_id"] in checked else "UNCHECKED"
        trial_lines.append(f"  {t['trial_id']} | {t['cancer_type']} | {status}")

    msg = (
        f"PATIENT: {patient['age']}yo {patient['gender']}, "
        f"{patient['cancer_type']}, stage {patient['stage']}\n"
        f"TRIALS:\n" + "\n".join(trial_lines) + "\n\n"
        f"Pick ONE JSON action:"
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg}
            ],
            temperature=0.0,
            max_tokens=60,
            timeout=8.0
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            action = json.loads(raw[start:end])
            if "type" in action:
                return action
    except Exception:
        pass
    return None

# -----------------------------------------------
# HEURISTIC FALLBACK AGENT
# -----------------------------------------------

def heuristic_action(observation: dict, step_history: list) -> dict:
    """Deterministic fallback: check unchecked trials, pick best, resolve."""
    trials = [t["trial_id"] for t in observation["available_trials"]]
    checked = observation["checked_trials"]
    selected = observation["selected_trial_id"]

    # If trial already selected -> resolve
    if selected is not None:
        return {"type": "resolve"}

    # Check unchecked trials one by one
    unchecked = [t for t in trials if t not in checked]
    if unchecked:
        return {"type": "check_criteria", "trial_id": unchecked[0]}

    # All checked — pick the best candidate from history
    best_trial = _pick_best_from_history(trials, step_history)
    return {"type": "select_trial", "trial_id": best_trial}


def _pick_best_from_history(trials: list, step_history: list) -> str:
    """Pick trial with inclusion_pass and no exclusion from check history."""
    for h in step_history:
        action = h.get("action", {})
        info = h.get("info", {})
        if action.get("type") == "check_criteria":
            tid = action.get("trial_id")
            if info.get("inclusion_pass") and not info.get("exclusion_triggered"):
                return tid
    # Fallback: first trial
    return trials[0] if trials else "unknown"

# -----------------------------------------------
# HYBRID AGENT: LLM (steps 1-2) + HEURISTIC
# -----------------------------------------------

def get_action(client: OpenAI | None, observation: dict,
               step_history: list, steps_taken: int) -> dict:
    """Pick action: LLM for first 2 steps, heuristic after."""
    trials = [t["trial_id"] for t in observation["available_trials"]]
    selected = observation["selected_trial_id"]
    steps_left = observation["max_steps"] - observation["steps_taken"]

    # SAFETY: force select+resolve when nearly out of steps
    if steps_left <= 2:
        if selected is None:
            best = _pick_best_from_history(trials, step_history)
            return {"type": "select_trial", "trial_id": best}
        return {"type": "resolve"}

    # SAFETY: if selected and running low, resolve
    if selected is not None and steps_left <= 4:
        return {"type": "resolve"}

    # LLM for first 2 steps only (if client available)
    if steps_taken <= 2 and client is not None:
        action = call_llm(client, observation)
        if action is not None:
            # Validate trial_id references exist
            if action.get("type") in ("check_criteria", "select_trial"):
                if action.get("trial_id") not in trials:
                    action["trial_id"] = trials[0]
            return action

    # Heuristic for all remaining steps
    return heuristic_action(observation, step_history)

# -----------------------------------------------
# RUN ONE TASK
# -----------------------------------------------

def run_task(client: OpenAI | None, task_id: str) -> dict:
    """Run hybrid agent on one task with strict output.
    
    Guarantees: [START] printed, at least 1 [STEP], [END] always printed.
    """
    start_time = time.time()
    step_history = []
    rewards_list = []
    done = False
    grade = 0.5
    steps = 0
    final_info = {}
    observation = None

    # --- RESET with error handling ---
    try:
        observation = env_reset(task_id)
    except Exception as e:
        # Reset failed — still emit valid [START] [STEP] [END]
        _log_start(task=task_id)
        _log_step(step=1, action_str='resolve()', reward=0.01,
                  done=True, error=f"reset_failed:{str(e)[:40]}")
        _log_end(success=False, steps=1, rewards=[0.01])
        return {"task_id": task_id, "grade": 0.01, "correct": False,
                "steps": 1, "done": True}

    # --- [START] after successful reset ---
    _log_start(task=task_id)

    # --- Main loop ---
    while not done and steps < MAX_STEPS:
        if time.time() - start_time > 4.0:
            break

        steps += 1
        action = get_action(client, observation, step_history, steps)

        try:
            result = env_step(action)
            observation = result["observation"]
            reward_val = result["reward"]["value"]
            done = result["done"]
            info = result["info"]

            step_history.append({
                "action": action,
                "reward": reward_val,
                "done": done,
                "info": info,
            })

            if done:
                grade = _clamp(info.get("grade", 0.5))
                final_info = info
                # Use clamped grade as the resolve reward
                rewards_list.append(grade)
                _log_step(step=steps, action_str=_fmt_action(action),
                          reward=grade, done=True, error="null")
                break
            else:
                rewards_list.append(reward_val)
                _log_step(step=steps, action_str=_fmt_action(action),
                          reward=reward_val, done=False, error="null")

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP_{e.response.status_code}"
            rewards_list.append(0.0)
            _log_step(step=steps, action_str=_fmt_action(action),
                      reward=0.0, done=False, error=error_msg)
            if e.response.status_code == 400:
                break

        except Exception as e:
            error_msg = str(e).replace(" ", "_")[:50]
            rewards_list.append(0.0)
            _log_step(step=steps, action_str=_fmt_action(action),
                      reward=0.0, done=False, error=error_msg)

    # --- Force finish if not done ---
    if not done:
        try:
            if observation and observation.get("selected_trial_id") is None:
                trials = [t["trial_id"] for t in observation["available_trials"]]
                best = _pick_best_from_history(trials, step_history)
                r = env_step({"type": "select_trial", "trial_id": best})
                steps += 1
                rewards_list.append(r["reward"]["value"])
                _log_step(step=steps,
                          action_str=f'select_trial("{best}")',
                          reward=r["reward"]["value"], done=r["done"], error="null")

            result = env_step({"type": "resolve"})
            steps += 1
            resolve_reward = result["reward"]["value"]
            grade = _clamp(result["info"].get("grade", 0.5))
            final_info = result["info"]
            done = True
            # Log and store clamped grade as the resolve reward
            rewards_list.append(grade)
            _log_step(step=steps, action_str='resolve()',
                      reward=grade, done=True, error="null")
        except Exception:
            grade = _clamp(0.0)

    # --- Guarantee at least 1 step ---
    if steps == 0:
        steps = 1
        rewards_list = [0.01]
        _log_step(step=1, action_str='resolve()', reward=0.01,
                  done=True, error="no_steps_executed")

    correct = final_info.get("correct", False)
    _log_end(success=correct, steps=steps, rewards=rewards_list)

    return {
        "task_id": task_id,
        "grade": grade,
        "correct": correct,
        "steps": steps,
        "done": done,
    }

# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main():
    # Ensure environment server is running
    try:
        _ensure_server()
    except Exception as e:
        # Can't reach server — emit minimal valid output for all 3 tasks
        for t in TASKS:
            _log_start(task=t)
            _log_step(step=1, action_str='resolve()', reward=0.01,
                      done=True, error="server_unavailable")
            _log_end(success=False, steps=1, rewards=[0.01])
        sys.exit(1)

    # Initialize LLM client (optional — works without it via heuristic)
    client = None
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception:
            pass

    # Run all 3 tasks (submission requires >= 3 tasks with graders)
    all_done = True
    for task_id in TASKS:
        result = run_task(client, task_id)
        if not result.get("done", False):
            all_done = False

    # Cleanup
    _stop_server()

    sys.exit(0 if all_done else 1)


if __name__ == "__main__":
    main()
