"""
Phase 3A: Hardcoded deterministic tasks for clinical trial matching.

Defines 3 tasks with fixed patients and trial pools:
- Task 1 (easy): single_match - obvious criteria
- Task 2 (medium): hidden_exclusion - traps with exclusion rules
- Task 3 (hard): ambiguous_match - complex with biomarker requirements
"""

from dataclasses import dataclass
from typing import Literal
from src.schemas.patient_schema import generate_random_patient
from src.schemas.trial_schema import generate_random_trial
from src.engine.eligibility_engine import (
    is_eligible,
    check_inclusion,
    check_exclusion,
    check_biomarkers,
    check_comorbidities
)


@dataclass
class TaskDefinition:
    """Definition of a deterministic task."""
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard", "expert"]
    description: str
    patient_seed: int
    trial_seeds: list[int]
    correct_trial_id: str
    num_trials: int
    mode: str = "single"
    patient_seeds: list = None
    correct_trial_ids: list = None
    
    def __post_init__(self):
        if self.patient_seeds is None:
            self.patient_seeds = []
        if self.correct_trial_ids is None:
            self.correct_trial_ids = []


# Task definitions with verified seeds
TASKS = [
    TaskDefinition(
        task_id="single_match",
        name="Single Match (Easy)",
        difficulty="easy",
        description=(
            "Pool has 3 trials. Patient is eligible for exactly 1 trial. "
            "The 2 wrong trials fail on obvious criteria (wrong cancer type). "
            "Agent should solve in 3-4 steps."
        ),
        patient_seed=1000,
        trial_seeds=[2002, 2003, 2004],
        correct_trial_id="TRIAL-LUNG-7944",
        num_trials=3
    ),
    TaskDefinition(
        task_id="hidden_exclusion",
        name="Hidden Exclusion (Medium)",
        difficulty="medium",
        description=(
            "Pool has 5 trials. Patient is eligible for exactly 1 trial. "
            "At least 2 trials pass inclusion but fail exclusion (traps). "
            "Agent must carefully check both inclusion AND exclusion criteria."
        ),
        patient_seed=1002,
        trial_seeds=[3025, 3026, 3027, 3028, 3029],
        correct_trial_id="TRIAL-COLON-8437",
        num_trials=5
    ),
    TaskDefinition(
        task_id="ambiguous_match",
        name="Ambiguous Match (Hard)",
        difficulty="hard",
        description=(
            "Pool has 7 trials. Patient is eligible for exactly 1 trial. "
            "At least 3 trials pass inclusion but fail exclusion (traps). "
            "At least 2 trials fail on biomarker requirements. "
            "The correct trial requires biomarker verification. "
            "Agent must investigate biomarkers AND check all criteria carefully."
        ),
        patient_seed=1012,
        trial_seeds=[4056, 4057, 4058, 4059, 4060, 4061, 4062],
        correct_trial_id="TRIAL-COLON-5245",
        num_trials=7
    )
]


# Extended tasks (not included in list_tasks() for backward compatibility)
EXTENDED_TASKS: list[TaskDefinition] = [
    TaskDefinition(
        task_id="multi_patient",
        name="Multi-Patient Matching (Expert)",
        difficulty="expert",
        description=(
            "Match 3 patients to their correct trials simultaneously. "
            "Trials shared across patients. Each patient eligible for exactly 1 different trial. "
            "Agent must manage multiple patient contexts at once."
        ),
        patient_seed=2011,
        trial_seeds=[5000, 5001, 5002, 5003, 5004],
        correct_trial_id="",
        num_trials=5,
        mode="multi",
        patient_seeds=[2011, 2020, 2026],
        correct_trial_ids=["TRIAL-LUNG-4570", "TRIAL-COLON-7254", "TRIAL-COLON-1441"]
    ),
    TaskDefinition(
        task_id="competing_trials",
        name="Competing Trials - Pick Best (Expert)",
        difficulty="expert",
        description=(
            "Patient is eligible for multiple trials. "
            "Agent must compare trial_score values and pick the highest-scoring eligible trial. "
            "Requires checking eligibility AND comparing quality scores."
        ),
        patient_seed=3041,
        trial_seeds=[6001, 6002, 6003, 6004, 6005],
        correct_trial_id="TRIAL-LUNG-4295",
        num_trials=5,
    ),
    TaskDefinition(
        task_id="contradictory_info",
        name="Contradictory Information (Expert)",
        difficulty="expert",
        description=(
            "Patient data contains contradictions in lab value trends. "
            "Agent must flag the contradiction using flag_contradiction action, "
            "then still select the correct eligible trial. "
            "Tests the agent's ability to detect data quality issues."
        ),
        patient_seed=4000,
        trial_seeds=[7040, 7041, 7042, 7043, 7044],
        correct_trial_id="TRIAL-COLON-5953",
        num_trials=5,
    ),
]


def get_task(task_id: str) -> TaskDefinition:
    """
    Get task definition by ID.
    
    Args:
        task_id: Task identifier
    
    Returns:
        TaskDefinition
    
    Raises:
        ValueError: If task_id not found
    """
    for task in TASKS:
        if task.task_id == task_id:
            return task
    for task in EXTENDED_TASKS:
        if task.task_id == task_id:
            return task
    all_ids = [t.task_id for t in TASKS] + [t.task_id for t in EXTENDED_TASKS]
    raise ValueError(f"Task '{task_id}' not found. Available: {all_ids}")


def list_tasks() -> list[TaskDefinition]:
    """
    List all available tasks.
    
    Returns:
        List of TaskDefinition objects
    """
    return TASKS + EXTENDED_TASKS


def verify_task_structure(task_id: str) -> bool:
    """
    Verify that a task has the correct structure.
    
    Prints detailed breakdown of patient, trials, and eligibility.
    
    Args:
        task_id: Task identifier
    
    Returns:
        bool: True if structure is valid, False otherwise
    """
    task = get_task(task_id)
    
    print(f"\n{'='*80}")
    print(f"TASK: {task.task_id} ({task.difficulty})")
    print(f"{'='*80}")
    
    # Generate patient
    patient = generate_random_patient(seed=task.patient_seed)
    print(f"\nPatient (seed={task.patient_seed}):")
    print(f"  ID: {patient.id}")
    print(f"  Age: {patient.age}")
    print(f"  Gender: {patient.gender}")
    print(f"  Cancer Type: {patient.cancer_type}")
    print(f"  Stage: {patient.stage}")
    print(f"  Biomarkers: EGFR={patient.biomarkers.EGFR}, ALK={patient.biomarkers.ALK}, PD_L1={patient.biomarkers.PD_L1}")
    print(f"  Lab Values: hb={patient.lab_values.hb:.2f}, wbc={patient.lab_values.wbc:.2f}, creatinine={patient.lab_values.creatinine:.2f}")
    print(f"  Comorbidities: {patient.comorbidities}")
    
    # Generate trials
    trials = [generate_random_trial(seed=s) for s in task.trial_seeds]
    
    print(f"\nTrials ({len(trials)} total):")
    print(f"{'Trial ID':<20} {'Cancer':<15} {'Inc':<6} {'Exc':<6} {'Bio':<6} {'Com':<6} {'Eligible':<10}")
    print("-" * 80)
    
    eligible_trials = []
    traps = []
    biomarker_fails = []
    
    for trial in trials:
        inc = check_inclusion(patient, trial)
        exc = check_exclusion(patient, trial)
        bio = check_biomarkers(patient, trial)
        com = check_comorbidities(patient, trial)
        elig = is_eligible(patient, trial)
        
        print(f"{trial.trial_id:<20} {trial.cancer_type:<15} {str(inc):<6} {str(exc):<6} {str(bio):<6} {str(com):<6} {str(elig):<10}")
        
        if elig:
            eligible_trials.append(trial.trial_id)
        
        if inc and exc:
            traps.append(trial.trial_id)
        
        if not bio:
            biomarker_fails.append(trial.trial_id)
    
    # Verify structure
    print(f"\n{'='*80}")
    print(f"STRUCTURE VERIFICATION")
    print(f"{'='*80}")
    
    valid = True
    
    # Check 1: Exactly 1 eligible trial
    print(f"\n1. Eligible trials: {len(eligible_trials)}")
    if len(eligible_trials) == 1:
        print(f"   ✅ PASS: Exactly 1 eligible trial")
        print(f"   Correct trial: {eligible_trials[0]}")
        if eligible_trials[0] == task.correct_trial_id:
            print(f"   ✅ PASS: Matches recorded correct_trial_id")
        else:
            print(f"   ❌ FAIL: Does not match recorded correct_trial_id ({task.correct_trial_id})")
            valid = False
    else:
        print(f"   ❌ FAIL: Expected 1, found {len(eligible_trials)}")
        valid = False
    
    # Task-specific checks
    if task.difficulty == "easy":
        # Check for obvious failures (wrong cancer type)
        wrong_cancer = sum(1 for t in trials if t.cancer_type != patient.cancer_type and t.trial_id not in eligible_trials)
        print(f"\n2. Wrong cancer type trials: {wrong_cancer}")
        if wrong_cancer >= 2:
            print(f"   ✅ PASS: At least 2 trials fail on obvious criteria")
        else:
            print(f"   ❌ FAIL: Expected at least 2 obvious failures")
            valid = False
    
    elif task.difficulty == "medium":
        # Check for traps (inclusion pass, exclusion triggered)
        print(f"\n2. Trap trials (inclusion pass, exclusion triggered): {len(traps)}")
        if len(traps) >= 2:
            print(f"   ✅ PASS: At least 2 trap trials")
            print(f"   Traps: {traps}")
        else:
            print(f"   ❌ FAIL: Expected at least 2 traps, found {len(traps)}")
            valid = False
    
    elif task.difficulty == "hard":
        # Check for traps
        print(f"\n2. Trap trials (inclusion pass, exclusion triggered): {len(traps)}")
        if len(traps) >= 3:
            print(f"   ✅ PASS: At least 3 trap trials")
        else:
            print(f"   ❌ FAIL: Expected at least 3 traps, found {len(traps)}")
            valid = False
        
        # Check for biomarker failures
        print(f"\n3. Biomarker failure trials: {len(biomarker_fails)}")
        if len(biomarker_fails) >= 2:
            print(f"   ✅ PASS: At least 2 biomarker failures")
        else:
            print(f"   ❌ FAIL: Expected at least 2 biomarker failures, found {len(biomarker_fails)}")
            valid = False
        
        # Check if correct trial has biomarker requirement
        if eligible_trials:
            correct_trial = next(t for t in trials if t.trial_id == eligible_trials[0])
            has_biomarker_req = (
                correct_trial.required_biomarkers.EGFR is not None or
                correct_trial.required_biomarkers.ALK is not None or
                correct_trial.required_biomarkers.PD_L1 is not None
            )
            print(f"\n4. Correct trial has biomarker requirement: {has_biomarker_req}")
            if has_biomarker_req:
                print(f"   ✅ PASS: Correct trial requires biomarker verification")
            else:
                print(f"   ❌ FAIL: Correct trial should have biomarker requirement")
                valid = False
    
    print(f"\n{'='*80}")
    if valid:
        print(f"✅ TASK {task.task_id}: STRUCTURE VALID")
    else:
        print(f"❌ TASK {task.task_id}: STRUCTURE INVALID")
    print(f"{'='*80}")
    
    return valid


def verify_all_tasks() -> bool:
    """
    Verify all tasks have correct structure.
    
    Returns:
        bool: True if all tasks are valid, False otherwise
    """
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "TASK VERIFICATION" + " "*36 + "║")
    print("╚" + "="*78 + "╝")
    
    results = []
    for task in TASKS:
        valid = verify_task_structure(task.task_id)
        results.append((task.task_id, valid))
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*30 + "SUMMARY" + " "*41 + "║")
    print("╚" + "="*78 + "╝\n")
    
    for task_id, valid in results:
        status = "✅ PASS" if valid else "❌ FAIL"
        print(f"{status}: {task_id}")
    
    all_valid = all(valid for _, valid in results)
    
    print(f"\n{'='*80}")
    if all_valid:
        print("✅ ALL TASKS VALID - Ready for Phase 3B")
    else:
        print("❌ SOME TASKS INVALID - Fix seeds before proceeding")
    print(f"{'='*80}\n")
    
    return all_valid
