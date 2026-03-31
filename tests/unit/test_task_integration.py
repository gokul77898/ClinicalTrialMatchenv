"""
Test suite for Phase 3C: Task integration with environment.

Validates task-based episode execution, grading, and reproducibility.
"""

import pytest
from src.environment import ClinicalTrialEnv
from src.models import Observation, Action
from src.tasks import get_task


class TestTaskIntegration:
    """Test suite for task integration."""
    
    def test_reset_with_task_id_works(self):
        """Test 1: reset with task_id works."""
        print("\n" + "="*80)
        print("TEST 1: reset with task_id works")
        print("="*80)
        
        env = ClinicalTrialEnv()
        obs = env.reset(task_id="single_match")
        
        print(f"\nObservation type: {type(obs)}")
        print(f"Steps taken: {obs.steps_taken}")
        print(f"Available trials: {len(obs.available_trials)}")
        
        assert isinstance(obs, Observation), f"Expected Observation, got {type(obs)}"
        assert obs.steps_taken == 0, f"Expected 0 steps, got {obs.steps_taken}"
        assert len(obs.available_trials) == 3, f"Expected 3 trials, got {len(obs.available_trials)}"
        
        print("✅ PASSED")
    
    def test_reset_with_task_id_loads_correct_trials(self):
        """Test 2: reset with task_id loads correct trials."""
        print("\n" + "="*80)
        print("TEST 2: reset with task_id loads correct trials")
        print("="*80)
        
        from src.schemas.trial_schema import generate_random_trial
        
        env = ClinicalTrialEnv()
        obs = env.reset(task_id="single_match")
        task = get_task("single_match")
        
        expected_trials = [generate_random_trial(seed=s) for s in task.trial_seeds]
        expected_ids = [t.trial_id for t in expected_trials]
        actual_ids = [t["trial_id"] for t in obs.available_trials]
        
        print(f"\nExpected trial IDs: {expected_ids}")
        print(f"Actual trial IDs: {actual_ids}")
        
        assert set(expected_ids) == set(actual_ids), \
            f"Trial IDs don't match: expected {expected_ids}, got {actual_ids}"
        
        print("✅ PASSED")
    
    def test_resolve_returns_grade_for_task_episode(self):
        """Test 3: resolve returns grade for task episode."""
        print("\n" + "="*80)
        print("TEST 3: resolve returns grade for task episode")
        print("="*80)
        
        env = ClinicalTrialEnv()
        obs = env.reset(task_id="single_match")
        task = get_task("single_match")
        
        env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
        obs, reward, done, info = env.step(Action(type="resolve"))
        
        print(f"\nGrade in info: {'grade' in info}")
        print(f"Grade value: {info.get('grade')}")
        print(f"Done: {done}")
        
        assert "grade" in info, "Grade not in info dict"
        assert info["grade"] is not None, "Grade is None"
        assert 0.0 <= info["grade"] <= 1.0, f"Grade {info['grade']} out of range [0.0, 1.0]"
        assert done == True, "Episode should be done"
        
        print("✅ PASSED")
    
    def test_correct_selection_gives_high_grade_on_easy(self):
        """Test 4: correct selection gives high grade on easy."""
        print("\n" + "="*80)
        print("TEST 4: correct selection gives high grade on easy")
        print("="*80)
        
        env = ClinicalTrialEnv()
        obs = env.reset(task_id="single_match")
        task = get_task("single_match")
        
        env.step(Action(type="check_criteria", trial_id=task.correct_trial_id))
        env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
        obs, reward, done, info = env.step(Action(type="resolve"))
        
        print(f"\nGrade: {info['grade']}")
        print(f"Correct trial: {task.correct_trial_id}")
        print(f"Selected trial: {info['selected_trial']}")
        
        assert info["grade"] >= 0.8, f"Expected grade >= 0.8, got {info['grade']}"
        
        print("✅ PASSED")
    
    def test_wrong_selection_gives_low_grade_on_easy(self):
        """Test 5: wrong selection gives low grade on easy."""
        print("\n" + "="*80)
        print("TEST 5: wrong selection gives low grade on easy")
        print("="*80)
        
        env = ClinicalTrialEnv()
        obs = env.reset(task_id="single_match")
        task = get_task("single_match")
        
        wrong_trial = [t["trial_id"] for t in obs.available_trials
                       if t["trial_id"] != task.correct_trial_id][0]
        
        print(f"\nCorrect trial: {task.correct_trial_id}")
        print(f"Selecting wrong trial: {wrong_trial}")
        
        env.step(Action(type="select_trial", trial_id=wrong_trial))
        obs, reward, done, info = env.step(Action(type="resolve"))
        
        print(f"Grade: {info['grade']}")
        
        assert info["grade"] < 0.5, f"Expected grade < 0.5, got {info['grade']}"
        
        print("✅ PASSED")
    
    def test_grade_is_none_when_no_task_id(self):
        """Test 6: grade is None when no task_id."""
        print("\n" + "="*80)
        print("TEST 6: grade is None when no task_id")
        print("="*80)
        
        env = ClinicalTrialEnv()
        env.reset(patient_seed=42, trial_seed=100)
        trial_id = env.state().available_trials[0]["trial_id"]
        
        env.step(Action(type="select_trial", trial_id=trial_id))
        obs, reward, done, info = env.step(Action(type="resolve"))
        
        print(f"\nGrade: {info.get('grade')}")
        
        assert info["grade"] is None, f"Expected grade=None, got {info['grade']}"
        
        print("✅ PASSED")
    
    def test_action_history_tracked_correctly(self):
        """Test 7: action history tracked correctly."""
        print("\n" + "="*80)
        print("TEST 7: action history tracked correctly")
        print("="*80)
        
        env = ClinicalTrialEnv()
        obs = env.reset(task_id="single_match")
        task = get_task("single_match")
        
        env.step(Action(type="investigate", field="age"))
        env.step(Action(type="check_criteria", trial_id=task.correct_trial_id))
        env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
        obs, reward, done, info = env.step(Action(type="resolve"))
        
        print(f"\nGrade: {info['grade']}")
        print(f"Steps taken: {obs.steps_taken}")
        
        assert info["grade"] is not None, "Grade should not be None"
        assert info["grade"] >= 0.8, f"Expected grade >= 0.8, got {info['grade']}"
        
        print("✅ PASSED")
    
    def test_all_3_tasks_run_without_error(self):
        """Test 8: all 3 tasks run without error."""
        print("\n" + "="*80)
        print("TEST 8: all 3 tasks run without error")
        print("="*80)
        
        env = ClinicalTrialEnv()
        
        for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
            print(f"\nTesting task: {task_id}")
            task = get_task(task_id)
            obs = env.reset(task_id=task_id)
            
            assert obs is not None, f"Observation is None for {task_id}"
            
            env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
            obs, reward, done, info = env.step(Action(type="resolve"))
            
            print(f"  Done: {done}")
            print(f"  Grade: {info['grade']}")
            
            assert done == True, f"Episode not done for {task_id}"
            assert "grade" in info, f"Grade not in info for {task_id}"
            assert 0.0 <= info["grade"] <= 1.0, f"Grade out of range for {task_id}"
        
        print("\n✅ PASSED")
    
    def test_episode_fully_reproducible(self):
        """Test 9: episode fully reproducible."""
        print("\n" + "="*80)
        print("TEST 9: episode fully reproducible")
        print("="*80)
        
        def run_episode(task_id):
            env = ClinicalTrialEnv()
            task = get_task(task_id)
            env.reset(task_id=task_id)
            env.step(Action(type="investigate", field="age"))
            env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
            _, _, _, info = env.step(Action(type="resolve"))
            return info["grade"]
        
        grade_1 = run_episode("single_match")
        grade_2 = run_episode("single_match")
        
        print(f"\nFirst run: {grade_1}")
        print(f"Second run: {grade_2}")
        
        assert grade_1 == grade_2, f"Grades differ: {grade_1} != {grade_2}"
        
        print("✅ PASSED")
    
    def test_medium_task_correct_selection_grades_high(self):
        """Test 10: medium task correct selection grades 0.85+."""
        print("\n" + "="*80)
        print("TEST 10: medium task correct selection grades 0.85+")
        print("="*80)
        
        env = ClinicalTrialEnv()
        task = get_task("hidden_exclusion")
        obs = env.reset(task_id="hidden_exclusion")
        
        # Check all trials
        for t in obs.available_trials:
            env.step(Action(type="check_criteria", trial_id=t["trial_id"]))
        
        # Investigate fields
        for field in ["age", "lab_values.hb", "lab_values.creatinine"]:
            env.step(Action(type="investigate", field=field))
        
        # Select correct trial
        env.step(Action(type="select_trial", trial_id=task.correct_trial_id))
        _, _, _, info = env.step(Action(type="resolve"))
        
        print(f"\nGrade: {info['grade']}")
        print(f"Steps taken: {obs.steps_taken}")
        
        assert info["grade"] >= 0.85, f"Expected grade >= 0.85, got {info['grade']}"
        
        print("✅ PASSED")


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "TASK INTEGRATION TEST SUITE" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    
    test_suite = TestTaskIntegration()
    
    tests = [
        ("reset with task_id works", test_suite.test_reset_with_task_id_works),
        ("reset with task_id loads correct trials", test_suite.test_reset_with_task_id_loads_correct_trials),
        ("resolve returns grade for task episode", test_suite.test_resolve_returns_grade_for_task_episode),
        ("correct selection gives high grade on easy", test_suite.test_correct_selection_gives_high_grade_on_easy),
        ("wrong selection gives low grade on easy", test_suite.test_wrong_selection_gives_low_grade_on_easy),
        ("grade is None when no task_id", test_suite.test_grade_is_none_when_no_task_id),
        ("action history tracked correctly", test_suite.test_action_history_tracked_correctly),
        ("all 3 tasks run without error", test_suite.test_all_3_tasks_run_without_error),
        ("episode fully reproducible", test_suite.test_episode_fully_reproducible),
        ("medium task correct selection grades 0.85+", test_suite.test_medium_task_correct_selection_grades_high),
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
