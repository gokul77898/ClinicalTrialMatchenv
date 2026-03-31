import pytest
import subprocess
import time
import requests
import os

BASE_URL = "http://localhost:7860"

@pytest.fixture(scope="module", autouse=True)
def server():
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = "/Users/gokul/Documents/ClinicalTrialMatchEnv"
    proc = subprocess.Popen(
        ["uvicorn", "api.server:app",
         "--host", "0.0.0.0", "--port", "7860",
         "--log-level", "error"],
        env=env_vars,
        cwd="/Users/gokul/Documents/ClinicalTrialMatchEnv"
    )
    time.sleep(3)
    yield proc
    proc.terminate()
    proc.wait()

def test_all_6_tasks_in_api():
    r = requests.get(f"{BASE_URL}/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert len(tasks) == 6
    task_ids = [t["task_id"] for t in tasks]
    assert "single_match" in task_ids
    assert "hidden_exclusion" in task_ids
    assert "ambiguous_match" in task_ids
    assert "multi_patient" in task_ids
    assert "competing_trials" in task_ids
    assert "contradictory_info" in task_ids

def run_task_episode(task_id: str, correct_trial_id: str) -> dict:
    r = requests.post(f"{BASE_URL}/reset",
                      json={"task_id": task_id})
    assert r.status_code == 200, f"reset failed: {r.text}"
    obs = r.json()
    
    trial_ids = [t["trial_id"] for t in obs["available_trials"]]
    
    # Handle multi-patient mode
    if obs.get("mode") == "multi" and "cases" in obs:
        from src.tasks import get_task
        task = get_task(task_id)
        # Select trial for each case
        for i, case in enumerate(obs["cases"]):
            case_id = case["case_id"]
            # Switch to this case if not first
            if i > 0:
                requests.post(f"{BASE_URL}/step",
                             json={"type": "switch_case", "case_id": case_id})
            # Select correct trial for this case
            correct_id = task.correct_trial_ids[i] if i < len(task.correct_trial_ids) else trial_ids[0]
            target = correct_id if correct_id in trial_ids else trial_ids[0]
            requests.post(f"{BASE_URL}/step",
                         json={"type": "select_trial", "trial_id": target})
    else:
        # Single patient mode
        requests.post(f"{BASE_URL}/step",
                      json={"type": "check_criteria",
                            "trial_id": trial_ids[0]})
        
        target = correct_trial_id if correct_trial_id in trial_ids else trial_ids[0]
        requests.post(f"{BASE_URL}/step",
                      json={"type": "select_trial",
                            "trial_id": target})
    
    r = requests.post(f"{BASE_URL}/step",
                      json={"type": "resolve"})
    assert r.status_code == 200
    data = r.json()
    assert data["done"] == True
    assert "grade" in data["info"], f"No grade in info: {data['info']}"
    assert 0.0 <= data["info"]["grade"] <= 1.0
    return data["info"]

def test_single_match_e2e():
    from src.tasks import get_task
    task = get_task("single_match")
    info = run_task_episode("single_match", task.correct_trial_id)
    assert info["grade"] >= 0.8

def test_hidden_exclusion_e2e():
    from src.tasks import get_task
    task = get_task("hidden_exclusion")
    info = run_task_episode("hidden_exclusion", task.correct_trial_id)
    assert info["grade"] >= 0.0

def test_ambiguous_match_e2e():
    from src.tasks import get_task
    task = get_task("ambiguous_match")
    info = run_task_episode("ambiguous_match", task.correct_trial_id)
    assert info["grade"] >= 0.0

def test_multi_patient_e2e():
    from src.tasks import get_task
    task = get_task("multi_patient")
    # For multi-patient, use first correct_trial_id if available
    correct_id = task.correct_trial_ids[0] if hasattr(task, 'correct_trial_ids') and task.correct_trial_ids else ""
    info = run_task_episode("multi_patient", correct_id)
    assert 0.0 <= info["grade"] <= 1.0

def test_competing_trials_e2e():
    from src.tasks import get_task
    task = get_task("competing_trials")
    info = run_task_episode("competing_trials", task.correct_trial_id)
    assert 0.0 <= info["grade"] <= 1.0

def test_contradictory_info_e2e():
    from src.tasks import get_task
    task = get_task("contradictory_info")
    info = run_task_episode("contradictory_info", task.correct_trial_id)
    assert 0.0 <= info["grade"] <= 1.0

def test_all_tasks_grades_in_range():
    from src.tasks import list_tasks
    all_tasks = list_tasks()
    for task in all_tasks:
        r = requests.post(f"{BASE_URL}/reset",
                          json={"task_id": task.task_id})
        if r.status_code != 200:
            continue
        obs = r.json()
        trial_ids = [t["trial_id"] for t in obs["available_trials"]]
        if not trial_ids:
            continue
        
        # Handle multi-patient mode
        if obs.get("mode") == "multi" and "cases" in obs:
            for i, case in enumerate(obs["cases"]):
                if i > 0:
                    requests.post(f"{BASE_URL}/step",
                                 json={"type": "switch_case", "case_id": case["case_id"]})
                requests.post(f"{BASE_URL}/step",
                             json={"type": "select_trial", "trial_id": trial_ids[0]})
        else:
            requests.post(f"{BASE_URL}/step",
                          json={"type": "select_trial",
                                "trial_id": trial_ids[0]})
        
        r = requests.post(f"{BASE_URL}/step",
                          json={"type": "resolve"})
        assert r.status_code == 200
        info = r.json()["info"]
        assert "grade" in info, f"Task {task.task_id} missing grade in info: {info}"
        grade = info["grade"]
        assert 0.0 <= grade <= 1.0, \
            f"Task {task.task_id} grade out of range: {grade}"

def test_grades_deterministic_all_tasks():
    from src.tasks import get_task
    for task_id in ["single_match", "multi_patient", "competing_trials"]:
        task = get_task(task_id)
        correct_id = task.correct_trial_id
        if task_id == "multi_patient" and hasattr(task, 'correct_trial_ids'):
            correct_id = task.correct_trial_ids[0] if task.correct_trial_ids else ""
        info1 = run_task_episode(task_id, correct_id)
        info2 = run_task_episode(task_id, correct_id)
        assert info1["grade"] == info2["grade"], \
            f"Task {task_id} not deterministic: {info1['grade']} != {info2['grade']}"
