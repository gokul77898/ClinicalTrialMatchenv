"""
Phase 3D: End-to-end validation of all 3 tasks.

Final gate test before Phase 4. Simulates 3 agent types against all tasks.
"""

import pytest
import random
from src.environment import ClinicalTrialEnv
from src.models import Action
from src.tasks import get_task, list_tasks


def run_perfect_agent(task_id: str) -> dict:
    """
    Run perfect agent that knows correct trial and investigates properly.
    
    Args:
        task_id: Task identifier
    
    Returns:
        dict: Results including grade, reward, steps, correct
    """
    env = ClinicalTrialEnv()
    task = get_task(task_id)
    obs = env.reset(task_id=task_id)
    
    if task_id == "single_match":
        # Easy task: minimal investigation (3 steps for full efficiency bonus)
        env.step(Action(type="check_criteria", trial_id=task.correct_trial_id))
        env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
        obs_final, reward, done, info = env.step(Action(type="resolve"))
    
    elif task_id == "hidden_exclusion":
        # Medium task: investigate key fields and check all trials
        env.step(Action(type="investigate", field="age"))
        env.step(Action(type="investigate", field="lab_values.hb"))
        env.step(Action(type="investigate", field="lab_values.creatinine"))
        
        for trial in obs.available_trials:
            env.step(Action(type="check_criteria", trial_id=trial["trial_id"]))
        
        env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
        obs_final, reward, done, info = env.step(Action(type="resolve"))
    
    elif task_id == "ambiguous_match":
        # Hard task: investigate all biomarkers, labs, and check all trials
        env.step(Action(type="investigate", field="age"))
        env.step(Action(type="investigate", field="biomarkers.EGFR"))
        env.step(Action(type="investigate", field="biomarkers.ALK"))
        env.step(Action(type="investigate", field="biomarkers.PD_L1"))
        env.step(Action(type="investigate", field="lab_values.hb"))
        env.step(Action(type="investigate", field="lab_values.creatinine"))
        
        for trial in obs.available_trials:
            env.step(Action(type="check_criteria", trial_id=trial["trial_id"]))
        
        env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
        obs_final, reward, done, info = env.step(Action(type="resolve"))
    
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    return {
        "task_id": task_id,
        "agent": "perfect",
        "grade": info["grade"],
        "reward": reward.value,
        "steps": obs_final.steps_taken,
        "correct": info["correct"]
    }


def run_random_agent(task_id: str) -> dict:
    """
    Run random agent that picks random trial without investigation.
    
    Args:
        task_id: Task identifier
    
    Returns:
        dict: Results including grade, reward, steps, correct
    """
    env = ClinicalTrialEnv()
    task = get_task(task_id)
    obs = env.reset(task_id=task_id)
    
    # Pick random trial (with seed for reproducibility)
    random.seed(999)
    random_trial_id = random.choice([t["trial_id"] for t in obs.available_trials])
    
    env.step(Action(type="select_trial", trial_id=random_trial_id))
    obs_final, reward, done, info = env.step(Action(type="resolve"))
    
    return {
        "task_id": task_id,
        "agent": "random",
        "grade": info["grade"],
        "reward": reward.value,
        "steps": obs_final.steps_taken,
        "correct": info["correct"]
    }


def run_systematic_agent(task_id: str) -> dict:
    """
    Run systematic agent that investigates all fields and checks all trials.
    
    Args:
        task_id: Task identifier
    
    Returns:
        dict: Results including grade, reward, steps, correct
    """
    env = ClinicalTrialEnv()
    task = get_task(task_id)
    obs = env.reset(task_id=task_id)
    
    # Investigate all fields
    env.step(Action(type="investigate", field="age"))
    env.step(Action(type="investigate", field="cancer_type"))
    env.step(Action(type="investigate", field="stage"))
    env.step(Action(type="investigate", field="biomarkers.EGFR"))
    env.step(Action(type="investigate", field="biomarkers.ALK"))
    env.step(Action(type="investigate", field="biomarkers.PD_L1"))
    env.step(Action(type="investigate", field="lab_values.hb"))
    env.step(Action(type="investigate", field="lab_values.wbc"))
    env.step(Action(type="investigate", field="lab_values.creatinine"))
    
    # Check all trials
    for trial in obs.available_trials:
        env.step(Action(type="check_criteria", trial_id=trial["trial_id"]))
    
    # Systematic agent picks correct trial (for fair comparison)
    env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
    obs_final, reward, done, info = env.step(Action(type="resolve"))
    
    return {
        "task_id": task_id,
        "agent": "systematic",
        "grade": info["grade"],
        "reward": reward.value,
        "steps": obs_final.steps_taken,
        "correct": info["correct"]
    }


class TestPhase3E2E:
    """End-to-end validation tests for Phase 3."""
    
    def test_perfect_agent_scores_on_all_tasks(self):
        """Test 1: Perfect agent scores on all tasks."""
        print("\n" + "="*80)
        print("TEST 1: Perfect agent scores on all tasks")
        print("="*80)
        
        results = {}
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            result = run_perfect_agent(task_id)
            results[task_id] = result
            
            print(f"\nTask: {task_id}")
            print(f"  Grade: {result['grade']:.2f}")
            print(f"  Steps: {result['steps']}")
            print(f"  Correct: {result['correct']}")
            
            assert result["correct"] == True, f"Perfect agent failed on {task_id}"
            assert result["grade"] >= 0.8, f"Perfect agent grade too low on {task_id}: {result['grade']}"
        
        print("\n✅ PASSED")
    
    def test_random_agent_scores_low(self):
        """Test 2: Random agent scores low."""
        print("\n" + "="*80)
        print("TEST 2: Random agent scores low")
        print("="*80)
        
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            result = run_random_agent(task_id)
            
            print(f"\nTask: {task_id}")
            print(f"  Grade: {result['grade']:.2f}")
            print(f"  Correct: {result['correct']}")
            
            assert result["grade"] <= 0.65, \
                f"Random agent grade too high on {task_id}: {result['grade']}"
        
        print("\n✅ PASSED")
    
    def test_perfect_agent_beats_random_agent(self):
        """Test 3: Perfect agent beats random agent on all tasks."""
        print("\n" + "="*80)
        print("TEST 3: Perfect agent beats random agent")
        print("="*80)
        
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            perfect = run_perfect_agent(task_id)
            random_r = run_random_agent(task_id)
            
            print(f"\nTask: {task_id}")
            print(f"  Perfect: {perfect['grade']:.2f}")
            print(f"  Random: {random_r['grade']:.2f}")
            
            assert perfect["grade"] > random_r["grade"], \
                f"Task {task_id}: perfect={perfect['grade']} <= random={random_r['grade']}"
        
        print("\n✅ PASSED")
    
    def test_grades_strictly_in_range(self):
        """Test 4: Grades strictly between 0.0 and 1.0."""
        print("\n" + "="*80)
        print("TEST 4: Grades strictly in range [0.0, 1.0]")
        print("="*80)
        
        count = 0
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            for agent_fn in [run_perfect_agent, run_random_agent, run_systematic_agent]:
                result = agent_fn(task_id)
                count += 1
                assert 0.0 <= result["grade"] <= 1.0, \
                    f"Grade out of range: {result['grade']}"
        
        print(f"\nChecked {count} grades, all in valid range")
        print("✅ PASSED")
    
    def test_all_episodes_fully_reproducible(self):
        """Test 5: All episodes fully reproducible."""
        print("\n" + "="*80)
        print("TEST 5: All episodes fully reproducible")
        print("="*80)
        
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            r1 = run_perfect_agent(task_id)
            r2 = run_perfect_agent(task_id)
            
            print(f"\nTask: {task_id}")
            print(f"  Run 1 grade: {r1['grade']:.2f}, steps: {r1['steps']}")
            print(f"  Run 2 grade: {r2['grade']:.2f}, steps: {r2['steps']}")
            
            assert r1["grade"] == r2["grade"], \
                f"Grades differ for {task_id}: {r1['grade']} != {r2['grade']}"
            assert r1["steps"] == r2["steps"], \
                f"Steps differ for {task_id}: {r1['steps']} != {r2['steps']}"
        
        print("\n✅ PASSED")
    
    def test_difficulty_progression_is_real(self):
        """Test 6: Difficulty progression is real."""
        print("\n" + "="*80)
        print("TEST 6: Difficulty progression is real")
        print("="*80)
        
        perfect_easy = run_perfect_agent("single_match")
        perfect_hard = run_perfect_agent("ambiguous_match")
        
        print(f"\nEasy task (single_match): {perfect_easy['steps']} steps")
        print(f"Hard task (ambiguous_match): {perfect_hard['steps']} steps")
        
        assert perfect_hard["steps"] > perfect_easy["steps"], \
            f"Hard task should require more steps: {perfect_hard['steps']} <= {perfect_easy['steps']}"
        
        print("✅ PASSED")
    
    def test_systematic_agent_scores(self):
        """Test 7: Systematic agent scores."""
        print("\n" + "="*80)
        print("TEST 7: Systematic agent scores")
        print("="*80)
        
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            result = run_systematic_agent(task_id)
            
            print(f"\nTask: {task_id}")
            print(f"  Grade: {result['grade']:.2f}")
            print(f"  Steps: {result['steps']}")
            print(f"  Correct: {result['correct']}")
            
            assert result["grade"] >= 0.5, \
                f"Systematic agent grade too low on {task_id}: {result['grade']}"
            assert result["correct"] == True, \
                f"Systematic agent should pick correct trial on {task_id}"
        
        print("\n✅ PASSED")
    
    def test_episode_does_not_leak_state(self):
        """Test 8: Episode does not leak state between runs."""
        print("\n" + "="*80)
        print("TEST 8: Episode does not leak state")
        print("="*80)
        
        r1 = run_perfect_agent("single_match")
        print(f"\nRun 1 (single_match): grade={r1['grade']:.2f}")
        
        run_perfect_agent("hidden_exclusion")
        print(f"Run 2 (hidden_exclusion): completed")
        
        r2 = run_perfect_agent("single_match")
        print(f"Run 3 (single_match): grade={r2['grade']:.2f}")
        
        assert r1["grade"] == r2["grade"], \
            f"State leaked between runs: {r1['grade']} != {r2['grade']}"
        
        print("\n✅ PASSED")
    
    def test_print_full_results_matrix(self):
        """Test 9: Print full results matrix."""
        print("\n" + "="*80)
        print("TEST 9: Print full results matrix")
        print("="*80)
        
        print("\n" + "="*80)
        print("AGENT RESULTS MATRIX")
        print("="*80)
        print(f"{'Task':<20} | {'Agent':<10} | {'Grade':<6} | {'Steps':<5} | {'Correct':<7}")
        print("="*80)
        
        results = []
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            for agent_fn, agent_name in [
                (run_perfect_agent, "perfect"),
                (run_random_agent, "random"),
                (run_systematic_agent, "systematic")
            ]:
                result = agent_fn(task_id)
                results.append(result)
                print(f"{task_id:<20} | {agent_name:<10} | {result['grade']:<6.2f} | {result['steps']:<5} | {str(result['correct']):<7}")
        
        print("="*80)
        
        # Assert all 9 cells have valid grades
        assert len(results) == 9, f"Expected 9 results, got {len(results)}"
        for result in results:
            assert result["grade"] is not None, "Grade is None"
            assert 0.0 <= result["grade"] <= 1.0, f"Grade out of range: {result['grade']}"
        
        print("\n✅ PASSED")
    
    def test_all_task_metadata_correct(self):
        """Test 10: All task metadata correct."""
        print("\n" + "="*80)
        print("TEST 10: All task metadata correct")
        print("="*80)
        
        tasks = list_tasks()
        
        print(f"\nTotal tasks: {len(tasks)}")
        assert len(tasks) == 6, f"Expected 6 tasks, got {len(tasks)}"
        
        difficulties = [t.difficulty for t in tasks]
        print(f"Difficulties: {difficulties}")
        
        assert "easy" in difficulties, "Missing 'easy' difficulty"
        assert "medium" in difficulties, "Missing 'medium' difficulty"
        assert "hard" in difficulties, "Missing 'hard' difficulty"
        assert "expert" in difficulties, "Missing 'expert' difficulty"
        
        for task in tasks:
            print(f"\nTask: {task.task_id}")
            print(f"  Difficulty: {task.difficulty}")
            print(f"  Correct trial: {task.correct_trial_id}")
            print(f"  Num trials: {task.num_trials}")
            print(f"  Trial seeds: {len(task.trial_seeds)}")
            
            assert task.task_id != "", "task_id is empty"
            # multi_patient uses correct_trial_ids (plural), not correct_trial_id
            if task.task_id != "multi_patient":
                assert task.correct_trial_id != "", f"correct_trial_id is empty for {task.task_id}"
            assert len(task.trial_seeds) == task.num_trials, \
                f"Trial seeds mismatch: {len(task.trial_seeds)} != {task.num_trials}"
        
        print("\n✅ PASSED")


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "PHASE 3 E2E TEST SUITE" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    
    test_suite = TestPhase3E2E()
    
    tests = [
        ("Perfect agent scores on all tasks", test_suite.test_perfect_agent_scores_on_all_tasks),
        ("Random agent scores low", test_suite.test_random_agent_scores_low),
        ("Perfect agent beats random agent", test_suite.test_perfect_agent_beats_random_agent),
        ("Grades strictly in range", test_suite.test_grades_strictly_in_range),
        ("All episodes fully reproducible", test_suite.test_all_episodes_fully_reproducible),
        ("Difficulty progression is real", test_suite.test_difficulty_progression_is_real),
        ("Systematic agent scores", test_suite.test_systematic_agent_scores),
        ("Episode does not leak state", test_suite.test_episode_does_not_leak_state),
        ("Print full results matrix", test_suite.test_print_full_results_matrix),
        ("All task metadata correct", test_suite.test_all_task_metadata_correct),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASSED", None))
        except AssertionError as e:
            results.append((test_name, "FAILED", str(e)))
        except Exception as e:
            results.append((test_name, "ERROR", str(e)))
    
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*30 + "TEST SUMMARY" + " "*37 + "║")
    print("╚" + "="*78 + "╝\n")
    
    for test_name, status, error in results:
        status_symbol = "✅" if status == "PASSED" else "❌"
        print(f"{status_symbol} {test_name}: {status}")
        if error:
            print(f"   Error: {error[:100]}...")
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {passed}/{total} tests passed")
    print(f"{'='*80}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
