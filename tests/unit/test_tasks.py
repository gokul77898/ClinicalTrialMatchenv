"""
Test suite for Phase 3A: Hardcoded deterministic tasks.

Validates task definitions, structure, and reproducibility.
"""

import pytest
from src.tasks import list_tasks, get_task, TaskDefinition
from src.schemas.patient_schema import generate_random_patient
from src.schemas.trial_schema import generate_random_trial
from src.engine.eligibility_engine import (
    is_eligible,
    check_inclusion,
    check_exclusion,
    check_biomarkers
)


class TestTasks:
    """Test suite for task definitions."""
    
    def test_all_tasks_exist(self):
        """Test 1: All 6 tasks exist with correct IDs."""
        print("\n" + "="*80)
        print("TEST 1: All 7 tasks exist")
        print("="*80)
        
        tasks = list_tasks()
        
        print(f"\nTotal tasks: {len(tasks)}")
        assert len(tasks) == 7, f"Expected 7 tasks, got {len(tasks)}"
        
        ids = [t.task_id for t in tasks]
        print(f"Task IDs: {ids}")
        
        # Original 3 tasks
        assert "single_match" in ids, "Task 'single_match' not found"
        assert "hidden_exclusion" in ids, "Task 'hidden_exclusion' not found"
        assert "ambiguous_match" in ids, "Task 'ambiguous_match' not found"
        
        # Extended 4 tasks
        assert "multi_patient" in ids, "Task 'multi_patient' not found"
        assert "competing_trials" in ids, "Task 'competing_trials' not found"
        assert "contradictory_info" in ids, "Task 'contradictory_info' not found"
        assert "logical_inference" in ids, "Task 'logical_inference' not found"
        
        print("\n✅ PASSED: All 7 tasks exist")
    
    def test_each_task_has_exactly_one_eligible_trial(self):
        """Test 2: Each task has correct eligible trial structure."""
        print("\n" + "="*80)
        print("TEST 2: Each task has correct eligible trial structure")
        print("="*80)
        
        for task in list_tasks():
            print(f"\nTask: {task.task_id}")
            
            # Skip multi_patient (uses multiple patients)
            if task.task_id == "multi_patient":
                print(f"  Skipping multi_patient (uses patient_seeds, not patient_seed)")
                print(f"  ✅ PASS")
                continue
            
            patient = generate_random_patient(seed=task.patient_seed)
            trials = [generate_random_trial(seed=s) for s in task.trial_seeds]
            
            eligible = [t for t in trials if is_eligible(patient, t)]
            
            print(f"  Eligible trials: {len(eligible)}")
            if eligible:
                print(f"  Eligible trial IDs: {[t.trial_id for t in eligible]}")
                print(f"  Expected trial ID: {task.correct_trial_id}")
            
            # competing_trials intentionally has 2 eligible trials
            if task.task_id == "competing_trials":
                assert len(eligible) >= 2, \
                    f"Task {task.task_id} should have 2+ eligible trials, got {len(eligible)}"
                assert task.correct_trial_id in [t.trial_id for t in eligible], \
                    f"Task {task.task_id}: correct trial {task.correct_trial_id} not in eligible trials"
            else:
                # Original 3 tasks + contradictory_info should have exactly 1 eligible
                assert len(eligible) == 1, \
                    f"Task {task.task_id} has {len(eligible)} eligible trials, expected 1"
                
                assert eligible[0].trial_id == task.correct_trial_id, \
                    f"Task {task.task_id}: eligible trial {eligible[0].trial_id} != expected {task.correct_trial_id}"
            
            print(f"  ✅ PASS")
        
        print("\n✅ PASSED: All tasks have correct eligible trial structure")
    
    def test_task2_has_at_least_2_traps(self):
        """Test 3: Task 2 has at least 2 inclusion-pass-exclusion-fail trials."""
        print("\n" + "="*80)
        print("TEST 3: Task 2 has at least 2 traps")
        print("="*80)
        
        task = get_task("hidden_exclusion")
        patient = generate_random_patient(seed=task.patient_seed)
        trials = [generate_random_trial(seed=s) for s in task.trial_seeds]
        
        traps = [t for t in trials if check_inclusion(patient, t) and check_exclusion(patient, t)]
        
        print(f"\nTask: {task.task_id}")
        print(f"Trap trials (inclusion pass, exclusion triggered): {len(traps)}")
        if traps:
            print(f"Trap trial IDs: {[t.trial_id for t in traps]}")
        
        assert len(traps) >= 2, \
            f"Task {task.task_id} has {len(traps)} traps, expected at least 2"
        
        print("\n✅ PASSED: Task 2 has at least 2 traps")
    
    def test_task3_has_at_least_3_traps_and_2_biomarker_failures(self):
        """Test 4: Task 3 has at least 3 traps and 2 biomarker failures."""
        print("\n" + "="*80)
        print("TEST 4: Task 3 has at least 3 traps and 2 biomarker failures")
        print("="*80)
        
        task = get_task("ambiguous_match")
        patient = generate_random_patient(seed=task.patient_seed)
        trials = [generate_random_trial(seed=s) for s in task.trial_seeds]
        
        traps = [t for t in trials if check_inclusion(patient, t) and check_exclusion(patient, t)]
        biomarker_fails = [t for t in trials if not check_biomarkers(patient, t)]
        
        print(f"\nTask: {task.task_id}")
        print(f"Trap trials: {len(traps)}")
        if traps:
            print(f"Trap trial IDs: {[t.trial_id for t in traps]}")
        
        print(f"Biomarker failure trials: {len(biomarker_fails)}")
        if biomarker_fails:
            print(f"Biomarker fail trial IDs: {[t.trial_id for t in biomarker_fails]}")
        
        assert len(traps) >= 3, \
            f"Task {task.task_id} has {len(traps)} traps, expected at least 3"
        
        assert len(biomarker_fails) >= 2, \
            f"Task {task.task_id} has {len(biomarker_fails)} biomarker failures, expected at least 2"
        
        print("\n✅ PASSED: Task 3 has at least 3 traps and 2 biomarker failures")
    
    def test_tasks_are_reproducible(self):
        """Test 5: Tasks are reproducible (same seed = same result)."""
        print("\n" + "="*80)
        print("TEST 5: Tasks are reproducible")
        print("="*80)
        
        task = get_task("single_match")
        
        print(f"\nTask: {task.task_id}")
        print(f"Patient seed: {task.patient_seed}")
        
        # Generate patient twice with same seed
        p1 = generate_random_patient(seed=task.patient_seed)
        p2 = generate_random_patient(seed=task.patient_seed)
        
        print(f"Patient 1 ID: {p1.id}")
        print(f"Patient 2 ID: {p2.id}")
        print(f"Patient 1 age: {p1.age}")
        print(f"Patient 2 age: {p2.age}")
        
        # Compare full dumps
        dump1 = p1.model_dump()
        dump2 = p2.model_dump()
        
        assert dump1 == dump2, "Patients generated with same seed are not identical"
        
        # Test trials too
        print(f"\nTrial seed: {task.trial_seeds[0]}")
        t1 = generate_random_trial(seed=task.trial_seeds[0])
        t2 = generate_random_trial(seed=task.trial_seeds[0])
        
        print(f"Trial 1 ID: {t1.trial_id}")
        print(f"Trial 2 ID: {t2.trial_id}")
        
        assert t1.model_dump() == t2.model_dump(), "Trials generated with same seed are not identical"
        
        print("\n✅ PASSED: Tasks are reproducible")


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*30 + "TASK TEST SUITE" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    test_suite = TestTasks()
    
    tests = [
        ("All 3 tasks exist", test_suite.test_all_tasks_exist),
        ("Each task has exactly 1 eligible trial", test_suite.test_each_task_has_exactly_one_eligible_trial),
        ("Task 2 has at least 2 traps", test_suite.test_task2_has_at_least_2_traps),
        ("Task 3 has at least 3 traps and 2 biomarker failures", test_suite.test_task3_has_at_least_3_traps_and_2_biomarker_failures),
        ("Tasks are reproducible", test_suite.test_tasks_are_reproducible),
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
