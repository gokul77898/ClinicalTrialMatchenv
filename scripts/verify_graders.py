from src.environment import ClinicalTrialEnv
from src.models import Action
from src.tasks import get_task
from src.graders import EpisodeHistory, grade_task

print("=" * 60)
print("GRADER FIX VERIFICATION")
print("=" * 60)

# -----------------------------------------------
# VERIFY FIX 1: No selection = 0.0
# -----------------------------------------------
print("\nFIX 1: No selection = 0.0")

episode_no_selection = EpisodeHistory(
    task_id="single_match",
    patient_seed=1002,
    trial_seeds=[2011, 2012, 2013],
    actions_taken=[],
    final_selected_trial_id=None,
    final_reward=-1.0,
    steps_taken=1,
    done=True,
    correct_trial_id="TRIAL-LUNG-5985"
)

score = grade_task("single_match", episode_no_selection)
print(f"  No selection score: {score}")
assert score == 0.0, f"FAIL: Expected 0.0, got {score}"
print("  ✅ PASS: No selection = 0.0")

score = grade_task("hidden_exclusion", EpisodeHistory(
    task_id="hidden_exclusion",
    patient_seed=1005,
    trial_seeds=[3031, 3032, 3033, 3034, 3035],
    actions_taken=[],
    final_selected_trial_id=None,
    final_reward=-1.0,
    steps_taken=1,
    done=True,
    correct_trial_id="TRIAL-COLON-8848"
))
assert score == 0.0, f"FAIL: Expected 0.0, got {score}"
print("  ✅ PASS: No selection on medium = 0.0")

score = grade_task("ambiguous_match", EpisodeHistory(
    task_id="ambiguous_match",
    patient_seed=1039,
    trial_seeds=[4021, 4022, 4023, 4024, 4025, 4026, 4027],
    actions_taken=[],
    final_selected_trial_id=None,
    final_reward=-1.0,
    steps_taken=1,
    done=True,
    correct_trial_id="TRIAL-LUNG-4344"
))
assert score == 0.0, f"FAIL: Expected 0.0, got {score}"
print("  ✅ PASS: No selection on hard = 0.0")

# -----------------------------------------------
# VERIFY FIX 2: Wrong selection gets no efficiency
# -----------------------------------------------
print("\nFIX 2: Wrong selection gets no efficiency bonus")

episode_wrong_fast = EpisodeHistory(
    task_id="single_match",
    patient_seed=1002,
    trial_seeds=[2011, 2012, 2013],
    actions_taken=[
        {"type": "check_criteria", "trial_id": "TRIAL-COLON-5757"},
        {"type": "select_trial", "trial_id": "TRIAL-COLON-5757"},
        {"type": "resolve"}
    ],
    final_selected_trial_id="TRIAL-COLON-5757",
    final_reward=-1.0,
    steps_taken=3,
    done=True,
    correct_trial_id="TRIAL-LUNG-5985"
)

score = grade_task("single_match", episode_wrong_fast)
print(f"  Wrong selection fast score: {score}")
assert score < 0.4, f"FAIL: Expected < 0.4, got {score}"
assert score >= 0.0, f"FAIL: Score negative: {score}"
print(f"  ✅ PASS: Wrong selection gets no efficiency (score={score})")

# -----------------------------------------------
# VERIFY FIX 3: Perfect agent scores 1.0 on easy
# -----------------------------------------------
print("\nFIX 3: Perfect agent scores 1.0 on easy")

env = ClinicalTrialEnv()
task = get_task("single_match")
obs = env.reset(task_id="single_match")

env.step(Action(type="check_criteria",
                trial_id=task.correct_trial_id))
env.step(Action(type="select_trial",
                trial_id=task.correct_trial_id))
_, _, _, info = env.step(Action(type="resolve"))

grade = info["grade"]
print(f"  Perfect agent grade: {grade}")
assert grade == 1.0, f"FAIL: Expected 1.0, got {grade}"
print(f"  ✅ PASS: Perfect agent scores 1.0 on easy")

# -----------------------------------------------
# VERIFY: Run all 3 agents on all 3 tasks
# -----------------------------------------------
print("\nFULL MATRIX VERIFICATION")
print("=" * 60)

def run_perfect_agent(task_id):
    env = ClinicalTrialEnv()
    task = get_task(task_id)
    obs = env.reset(task_id=task_id)
    
    if task_id == "single_match":
        env.step(Action(type="check_criteria",
                        trial_id=task.correct_trial_id))
        env.step(Action(type="select_trial",
                        trial_id=task.correct_trial_id))
        _, _, _, info = env.step(Action(type="resolve"))
    
    elif task_id == "hidden_exclusion":
        env.step(Action(type="investigate", field="age"))
        env.step(Action(type="investigate", field="lab_values.hb"))
        env.step(Action(type="investigate", field="lab_values.creatinine"))
        for t in obs.available_trials:
            env.step(Action(type="check_criteria",
                            trial_id=t["trial_id"]))
        env.step(Action(type="select_trial",
                        trial_id=task.correct_trial_id))
        _, _, _, info = env.step(Action(type="resolve"))
    
    elif task_id == "ambiguous_match":
        env.step(Action(type="investigate", field="age"))
        env.step(Action(type="investigate", field="biomarkers.EGFR"))
        env.step(Action(type="investigate", field="biomarkers.ALK"))
        env.step(Action(type="investigate", field="biomarkers.PD_L1"))
        env.step(Action(type="investigate", field="lab_values.hb"))
        env.step(Action(type="investigate", field="lab_values.creatinine"))
        for t in obs.available_trials:
            env.step(Action(type="check_criteria",
                            trial_id=t["trial_id"]))
        env.step(Action(type="select_trial",
                        trial_id=task.correct_trial_id))
        _, _, _, info = env.step(Action(type="resolve"))
    
    return info["grade"]

def run_random_agent(task_id):
    import random
    random.seed(999)
    env = ClinicalTrialEnv()
    task = get_task(task_id)
    obs = env.reset(task_id=task_id)
    wrong_trials = [t["trial_id"] for t in obs.available_trials
                    if t["trial_id"] != task.correct_trial_id]
    wrong_trial = wrong_trials[0]
    env.step(Action(type="select_trial", trial_id=wrong_trial))
    _, _, _, info = env.step(Action(type="resolve"))
    return info["grade"]

print(f"{'Task':<20} {'Agent':<12} {'Grade':<8}")
print("-" * 42)

expected = {
    "single_match": {"perfect": 1.0, "random_max": 0.3},
    "hidden_exclusion": {"perfect_min": 0.85, "random_max": 0.4},
    "ambiguous_match": {"perfect_min": 0.85, "random_max": 0.4},
}

all_passed = True

for task_id in ["single_match", "hidden_exclusion", "ambiguous_match"]:
    pg = run_perfect_agent(task_id)
    rg = run_random_agent(task_id)
    
    print(f"{task_id:<20} {'perfect':<12} {pg:<8}")
    print(f"{task_id:<20} {'random':<12} {rg:<8}")
    
    if task_id == "single_match":
        if pg != 1.0:
            print(f"  ❌ FAIL: perfect should be 1.0, got {pg}")
            all_passed = False
        else:
            print(f"  ✅ perfect = 1.0")
        if rg > 0.3:
            print(f"  ❌ FAIL: random should be <= 0.3, got {rg}")
            all_passed = False
        else:
            print(f"  ✅ random <= 0.3")
    else:
        if pg < 0.85:
            print(f"  ❌ FAIL: perfect should be >= 0.85, got {pg}")
            all_passed = False
        else:
            print(f"  ✅ perfect >= 0.85")
        if rg > 0.4:
            print(f"  ❌ FAIL: random should be <= 0.4, got {rg}")
            all_passed = False
        else:
            print(f"  ✅ random <= 0.4")
    print()

print("=" * 60)
if all_passed:
    print("✅ ALL FIXES VERIFIED - Ready for Phase 4")
else:
    print("❌ SOME FIXES FAILED - Do not proceed to Phase 4")
print("=" * 60)
