"""
Phase 4: API integration tests for ClinicalTrialMatchEnv FastAPI server.

Tests require the server to be running on port 7860.
Server is started automatically via fixture.
"""

import pytest
import requests
import subprocess
import time
import os

BASE_URL = "http://localhost:7860"


@pytest.fixture(scope="module", autouse=True)
def server():
    """Start the FastAPI server for testing."""
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = "/Users/gokul/Documents/ClinicalTrialMatchEnv"
    proc = subprocess.Popen(
        ["uvicorn", "api.server:app",
         "--host", "0.0.0.0", "--port", "7860"],
        env=env_vars,
        cwd="/Users/gokul/Documents/ClinicalTrialMatchEnv"
    )
    time.sleep(3)
    yield proc
    proc.terminate()
    proc.wait()


class TestAPI:
    """API integration test suite."""

    def test_root_returns_200(self):
        """Test 1: GET / returns 200."""
        print("\n" + "="*80)
        print("TEST 1: GET / returns 200")
        print("="*80)

        r = requests.get(f"{BASE_URL}/")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data
        assert data["name"] == "ClinicalTrialMatchEnv"

        print(f"  Status: {r.status_code}")
        print(f"  Name: {data['name']}")
        print("✅ PASSED")

    def test_health_returns_ok(self):
        """Test 2: GET /health returns ok."""
        print("\n" + "="*80)
        print("TEST 2: GET /health returns ok")
        print("="*80)

        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.json()}")
        print("✅ PASSED")

    def test_reset_returns_observation(self):
        """Test 3: POST /reset returns observation."""
        print("\n" + "="*80)
        print("TEST 3: POST /reset returns observation")
        print("="*80)

        r = requests.post(f"{BASE_URL}/reset", json={})
        assert r.status_code == 200
        data = r.json()
        assert "patient" in data
        assert "available_trials" in data
        assert data["steps_taken"] == 0
        assert data["done"] == False

        print(f"  Status: {r.status_code}")
        print(f"  Steps: {data['steps_taken']}")
        print(f"  Trials: {len(data['available_trials'])}")
        print("✅ PASSED")

    def test_reset_with_task_id(self):
        """Test 4: POST /reset with task_id."""
        print("\n" + "="*80)
        print("TEST 4: POST /reset with task_id")
        print("="*80)

        r = requests.post(f"{BASE_URL}/reset",
                          json={"task_id": "single_match"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["available_trials"]) == 3

        print(f"  Status: {r.status_code}")
        print(f"  Trials: {len(data['available_trials'])}")
        print("✅ PASSED")

    def test_step_investigate(self):
        """Test 5: POST /step investigate."""
        print("\n" + "="*80)
        print("TEST 5: POST /step investigate")
        print("="*80)

        requests.post(f"{BASE_URL}/reset", json={})
        r = requests.post(f"{BASE_URL}/step",
                          json={"type": "investigate", "field": "age"})
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        assert data["done"] == False

        print(f"  Status: {r.status_code}")
        print(f"  Done: {data['done']}")
        print(f"  Reward: {data['reward']['value']}")
        print("✅ PASSED")

    def test_state_returns_observation(self):
        """Test 6: GET /state returns observation."""
        print("\n" + "="*80)
        print("TEST 6: GET /state returns observation")
        print("="*80)

        requests.post(f"{BASE_URL}/reset", json={})
        r = requests.get(f"{BASE_URL}/state")
        assert r.status_code == 200
        data = r.json()
        assert "patient" in data
        assert "steps_taken" in data

        print(f"  Status: {r.status_code}")
        print(f"  Steps: {data['steps_taken']}")
        print("✅ PASSED")

    def test_tasks_returns_3_tasks(self):
        """Test 7: GET /tasks returns 6 tasks."""
        print("\n" + "="*80)
        print("TEST 7: GET /tasks returns 6 tasks")
        print("="*80)

        r = requests.get(f"{BASE_URL}/tasks")
        assert r.status_code == 200
        data = r.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 7
        task_ids = [t["task_id"] for t in data["tasks"]]
        # Original 3 tasks
        assert "single_match" in task_ids
        assert "hidden_exclusion" in task_ids
        assert "ambiguous_match" in task_ids
        # Extended 4 tasks
        assert "multi_patient" in task_ids
        assert "competing_trials" in task_ids
        assert "contradictory_info" in task_ids
        assert "logical_inference" in task_ids

        print(f"  Status: {r.status_code}")
        print(f"  Tasks: {task_ids}")
        print("✅ PASSED")

    def test_full_episode_with_grade(self):
        """Test 8: Full episode with grade."""
        print("\n" + "="*80)
        print("TEST 8: Full episode with grade")
        print("="*80)

        requests.post(f"{BASE_URL}/reset",
                      json={"task_id": "single_match"})
        requests.post(f"{BASE_URL}/step",
                      json={"type": "check_criteria",
                            "trial_id": "TRIAL-COLON-8837"})
        requests.post(f"{BASE_URL}/step",
                      json={"type": "select_trial",
                            "trial_id": "TRIAL-COLON-8837"})
        r = requests.post(f"{BASE_URL}/step",
                          json={"type": "resolve"})
        assert r.status_code == 200
        data = r.json()
        assert data["done"] == True
        assert "grade" in data["info"]
        assert data["info"]["grade"] == 1.0

        print(f"  Status: {r.status_code}")
        print(f"  Done: {data['done']}")
        print(f"  Grade: {data['info']['grade']}")
        print("✅ PASSED")

    def test_step_after_done_returns_400(self):
        """Test 9: Step after done returns 400."""
        print("\n" + "="*80)
        print("TEST 9: Step after done returns 400")
        print("="*80)

        # Episode is done from test 8
        r = requests.post(f"{BASE_URL}/step",
                          json={"type": "investigate", "field": "age"})
        assert r.status_code == 400

        print(f"  Status: {r.status_code}")
        print("✅ PASSED")

    def test_reset_clears_state(self):
        """Test 10: Reset clears state."""
        print("\n" + "="*80)
        print("TEST 10: Reset clears state")
        print("="*80)

        requests.post(f"{BASE_URL}/reset", json={})
        r = requests.get(f"{BASE_URL}/state")
        data = r.json()
        assert data["steps_taken"] == 0
        assert data["done"] == False

        print(f"  Status: {r.status_code}")
        print(f"  Steps: {data['steps_taken']}")
        print(f"  Done: {data['done']}")
        print("✅ PASSED")
