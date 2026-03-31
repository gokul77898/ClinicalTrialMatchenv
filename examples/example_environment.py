"""
Example usage of ClinicalTrialEnv with action system and reward progression.

Demonstrates:
- Environment reset
- investigate() actions
- check_criteria() actions
- select_trial() action
- resolve() action
- Reward accumulation
"""

from src.environment import ClinicalTrialEnv


def print_step_result(step_num: int, action: dict, result: dict):
    """Print formatted step result."""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {action['type'].upper()}")
    print(f"{'='*80}")
    print(f"Action: {action}")
    print(f"Reward: {result['reward']:+.2f}")
    print(f"Total Reward: {result['total_reward']:.2f}")
    print(f"Steps Taken: {result['steps_taken']}/{result.get('steps_remaining', 0) + result['steps_taken']}")
    print(f"Done: {result['done']}")
    print(f"Info: {result['info']}")


def example_successful_run():
    """Example of a successful trial matching episode."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "EXAMPLE 1: SUCCESSFUL MATCHING" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    
    env = ClinicalTrialEnv(num_trials=3, max_steps=20)
    
    obs = env.reset(patient_seed=42, trial_seed=100)
    print(f"\n🔄 RESET")
    print(f"Patient ID: {obs['patient_id']}")
    print(f"Available Trials: {obs['available_trials']}")
    print(f"Max Steps: {obs['steps_remaining']}")
    
    patient_summary = env.get_patient_summary()
    print(f"\nPatient Cancer Type: {patient_summary['cancer_type']}")
    
    step_num = 1
    
    result = env.step({"type": "investigate", "field": "age"})
    print_step_result(step_num, {"type": "investigate", "field": "age"}, result)
    step_num += 1
    
    result = env.step({"type": "investigate", "field": "biomarkers.EGFR"})
    print_step_result(step_num, {"type": "investigate", "field": "biomarkers.EGFR"}, result)
    step_num += 1
    
    result = env.step({"type": "investigate", "field": "lab_values.hb"})
    print_step_result(step_num, {"type": "investigate", "field": "lab_values.hb"}, result)
    step_num += 1
    
    trial_id = obs['available_trials'][0]
    result = env.step({"type": "check_criteria", "trial_id": trial_id})
    print_step_result(step_num, {"type": "check_criteria", "trial_id": trial_id}, result)
    step_num += 1
    
    result = env.step({"type": "select_trial", "trial_id": trial_id})
    print_step_result(step_num, {"type": "select_trial", "trial_id": trial_id}, result)
    step_num += 1
    
    result = env.step({"type": "resolve"})
    print_step_result(step_num, {"type": "resolve"}, result)
    
    print(f"\n{'='*80}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*80}")
    print(f"Final Total Reward: {result['total_reward']:.2f}")
    print(f"Total Steps: {result['steps_taken']}")
    print(f"Outcome: {'✅ SUCCESS' if result['info'].get('correct', False) else '❌ FAILURE'}")


def example_efficient_run():
    """Example of an efficient run with bonus."""
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "EXAMPLE 2: EFFICIENT MATCHING" + " "*29 + "║")
    print("╚" + "="*78 + "╝")
    
    env = ClinicalTrialEnv(num_trials=3, max_steps=20)
    
    obs = env.reset(patient_seed=100, trial_seed=200)
    print(f"\n🔄 RESET")
    print(f"Patient ID: {obs['patient_id']}")
    print(f"Available Trials: {obs['available_trials']}")
    
    step_num = 1
    
    trial_id = obs['available_trials'][0]
    result = env.step({"type": "check_criteria", "trial_id": trial_id})
    print_step_result(step_num, {"type": "check_criteria", "trial_id": trial_id}, result)
    step_num += 1
    
    result = env.step({"type": "select_trial", "trial_id": trial_id})
    print_step_result(step_num, {"type": "select_trial", "trial_id": trial_id}, result)
    step_num += 1
    
    result = env.step({"type": "resolve"})
    print_step_result(step_num, {"type": "resolve"}, result)
    
    print(f"\n{'='*80}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*80}")
    print(f"Final Total Reward: {result['total_reward']:.2f}")
    print(f"Total Steps: {result['steps_taken']}")
    print(f"Efficiency Bonus: {'✅ YES (+0.1)' if result['steps_taken'] <= 5 else '❌ NO'}")
    print(f"Outcome: {'✅ SUCCESS' if result['info'].get('correct', False) else '❌ FAILURE'}")


def example_penalty_cases():
    """Example demonstrating penalty cases."""
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "EXAMPLE 3: PENALTY CASES" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    env = ClinicalTrialEnv(num_trials=3, max_steps=20)
    
    obs = env.reset(patient_seed=150, trial_seed=250)
    print(f"\n🔄 RESET")
    
    step_num = 1
    
    print(f"\n{'='*80}")
    print(f"TEST 1: Invalid Field")
    print(f"{'='*80}")
    result = env.step({"type": "investigate", "field": "invalid_field"})
    print(f"Action: investigate(invalid_field)")
    print(f"Reward: {result['reward']:+.2f} (Penalty for invalid field)")
    print(f"Info: {result['info']}")
    step_num += 1
    
    print(f"\n{'='*80}")
    print(f"TEST 2: Repeated Investigation")
    print(f"{'='*80}")
    result = env.step({"type": "investigate", "field": "age"})
    print(f"Action 1: investigate(age)")
    print(f"Reward: {result['reward']:+.2f}")
    step_num += 1
    
    result = env.step({"type": "investigate", "field": "age"})
    print(f"Action 2: investigate(age) [REPEAT]")
    print(f"Reward: {result['reward']:+.2f} (Penalty for unnecessary action)")
    print(f"Info: {result['info']}")
    step_num += 1
    
    print(f"\n{'='*80}")
    print(f"TEST 3: Invalid Action Format")
    print(f"{'='*80}")
    result = env.step({"type": "unknown_action"})
    print(f"Action: unknown_action")
    print(f"Reward: {result['reward']:+.2f} (Penalty for invalid action)")
    print(f"Info: {result['info']}")
    step_num += 1
    
    print(f"\n{'='*80}")
    print(f"TEST 4: Wrong Trial Selection")
    print(f"{'='*80}")
    
    trial_id = obs['available_trials'][2]
    result = env.step({"type": "select_trial", "trial_id": trial_id})
    print(f"Action: select_trial({trial_id})")
    print(f"Reward: {result['reward']:+.2f}")
    step_num += 1
    
    result = env.step({"type": "resolve"})
    print(f"Action: resolve()")
    print(f"Reward: {result['reward']:+.2f} (CRITICAL penalty if wrong)")
    print(f"Info: {result['info']}")
    
    print(f"\n{'='*80}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*80}")
    print(f"Final Total Reward: {result['total_reward']:.2f}")
    print(f"Total Steps: {result['steps_taken']}")


def example_reward_breakdown():
    """Detailed reward breakdown example."""
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "EXAMPLE 4: REWARD BREAKDOWN" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    
    env = ClinicalTrialEnv(num_trials=3, max_steps=20)
    
    obs = env.reset(patient_seed=200, trial_seed=300)
    print(f"\n🔄 RESET - Starting Total Reward: 0.00")
    
    rewards = []
    
    print(f"\n{'='*80}")
    print(f"REWARD PROGRESSION")
    print(f"{'='*80}")
    
    result = env.step({"type": "investigate", "field": "age"})
    rewards.append(result['reward'])
    print(f"Step 1: investigate(age)")
    print(f"  Reward: {result['reward']:+.2f} (neutral for valid field)")
    print(f"  Running Total: {result['total_reward']:.2f}")
    
    result = env.step({"type": "investigate", "field": "biomarkers.PD_L1"})
    rewards.append(result['reward'])
    print(f"\nStep 2: investigate(biomarkers.PD_L1)")
    print(f"  Reward: {result['reward']:+.2f} (neutral for valid field)")
    print(f"  Running Total: {result['total_reward']:.2f}")
    
    trial_id = obs['available_trials'][0]
    result = env.step({"type": "check_criteria", "trial_id": trial_id})
    rewards.append(result['reward'])
    print(f"\nStep 3: check_criteria({trial_id})")
    print(f"  Reward: {result['reward']:+.2f} (+0.2 inclusion + 0.2 exclusion)")
    print(f"  Running Total: {result['total_reward']:.2f}")
    
    result = env.step({"type": "select_trial", "trial_id": trial_id})
    rewards.append(result['reward'])
    print(f"\nStep 4: select_trial({trial_id})")
    print(f"  Reward: {result['reward']:+.2f} (no immediate reward)")
    print(f"  Running Total: {result['total_reward']:.2f}")
    
    result = env.step({"type": "resolve"})
    rewards.append(result['reward'])
    print(f"\nStep 5: resolve()")
    print(f"  Base Reward: {0.3 if result['info'].get('correct') else -0.3:+.2f} ({'correct' if result['info'].get('correct') else 'wrong'} selection)")
    print(f"  Efficiency Bonus: {'+0.1' if result['steps_taken'] <= 5 else '0.0'} (steps <= 5)")
    print(f"  Total Step Reward: {result['reward']:+.2f}")
    print(f"  Running Total: {result['total_reward']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"FINAL BREAKDOWN")
    print(f"{'='*80}")
    for i, r in enumerate(rewards, 1):
        print(f"Step {i}: {r:+.2f}")
    print(f"{'='*80}")
    print(f"Total: {sum(rewards):.2f}")
    print(f"Outcome: {'✅ SUCCESS' if result['info'].get('correct', False) else '❌ FAILURE'}")


def main():
    """Run all examples."""
    example_successful_run()
    example_efficient_run()
    example_penalty_cases()
    example_reward_breakdown()
    
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "EXAMPLES COMPLETE" + " "*36 + "║")
    print("╚" + "="*78 + "╝\n")


if __name__ == "__main__":
    main()
