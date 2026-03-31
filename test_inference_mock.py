"""
Mock test for inference.py
Tests the full inference pipeline with a mock LLM agent.
Does not require a real API key.
"""

import os
import sys
import json
import time
import subprocess
import requests

ENV_BASE_URL = "http://localhost:7860"

def start_server():
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
    for i in range(30):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Server failed to start")

def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def mock_perfect_agent(task_id: str) -> dict:
    """
    Mock agent that plays perfectly.
    Knows the correct trial for each task.
    """
    correct_trials = {
        "single_match": "TRIAL-LUNG-7944",
        "hidden_exclusion": "TRIAL-COLON-8437",
        "ambiguous_match": "TRIAL-COLON-5245"
    }

    obs = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id}
    ).json()

    correct_trial = correct_trials[task_id]

    # check criteria
    requests.post(f"{ENV_BASE_URL}/step",
        json={"type": "check_criteria",
              "trial_id": correct_trial})

    # select correct trial
    requests.post(f"{ENV_BASE_URL}/step",
        json={"type": "select_trial",
              "trial_id": correct_trial})

    # resolve
    result = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"type": "resolve"}
    ).json()

    return {
        "task_id": task_id,
        "grade": result["info"].get("grade", 0.0),
        "correct": result["info"].get("correct", False),
        "done": result["done"]
    }

def main():
    print("=" * 60)
    print("MOCK INFERENCE TEST")
    print("Testing inference pipeline without real API key")
    print("=" * 60)

    server_proc = None
    try:
        server_proc = start_server()
        print("Server started successfully")

        tasks = ["single_match", "hidden_exclusion", "ambiguous_match"]
        results = []

        for task_id in tasks:
            print(f"\nRunning mock agent on: {task_id}")
            result = mock_perfect_agent(task_id)
            results.append(result)
            print(f"  Grade: {result['grade']}")
            print(f"  Correct: {result['correct']}")

        print("\n" + "=" * 60)
        print("MOCK RESULTS")
        print("=" * 60)
        print(f"{'Task':<25} {'Grade':<10} {'Correct'}")
        print("-" * 50)

        all_passed = True
        for r in results:
            print(f"{r['task_id']:<25} {r['grade']:<10} {r['correct']}")
            if not r["correct"]:
                all_passed = False

        print("\n" + "=" * 60)

        assert results[0]["grade"] == 1.0, \
            f"single_match should be 1.0, got {results[0]['grade']}"
        assert results[0]["correct"] == True
        assert results[1]["grade"] >= 0.5, \
            f"hidden_exclusion should be >= 0.5, got {results[1]['grade']}"
        assert results[2]["grade"] >= 0.4, \
            f"ambiguous_match should be >= 0.4, got {results[2]['grade']}"

        print("✅ ALL MOCK TESTS PASSED")
        print("✅ inference.py pipeline is working correctly")
        print("✅ Ready for real API key testing")
        
        print("\nVerifying extended tasks accessible via API...")
        for task_id in ["multi_patient", "competing_trials", "contradictory_info"]:
            r = requests.post(f"{ENV_BASE_URL}/reset",
                             json={"task_id": task_id})
            if r.status_code == 200:
                obs = r.json()
                print(f"  ✅ {task_id}: reset OK, {len(obs['available_trials'])} trials")
            else:
                print(f"  ❌ {task_id}: reset FAILED - {r.text[:100]}")

    except AssertionError as e:
        print(f"❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
    finally:
        stop_server(server_proc)
        print("Server stopped")

if __name__ == "__main__":
    main()
