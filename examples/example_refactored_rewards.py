"""
Example demonstrating refactored reward system with decision dominance.

Key Properties:
1. Correct decision → strongly positive (~1.0+)
2. Wrong decision → clearly negative (<0)
3. Spam actions → net negative (reward hacking impossible)
"""

from src.environment import ClinicalTrialEnv


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + f" {title:^76} " + "║")
    print("╚" + "="*78 + "╝")


def print_step(step_num: int, action: dict, result: dict):
    """Print step details."""
    print(f"\nStep {step_num}: {action['type']}")
    print(f"  Reward: {result['reward']:+.2f}")
    print(f"  Total: {result['total_reward']:.2f}")
    print(f"  Info: {result['info'].get('message', '')}")


def example_correct_decision():
    """Example 1: Correct decision → strongly positive."""
    print_header("EXAMPLE 1: CORRECT DECISION (Strong Positive)")
    
    env = ClinicalTrialEnv(num_trials=5, max_steps=20)
    obs = env.reset(patient_seed=42, trial_seed=100)
    
    print(f"\n🔄 RESET")
    print(f"Available Trials: {obs['available_trials']}")
    
    step_num = 1
    
    # Investigate a few fields
    result = env.step({"type": "investigate", "field": "age"})
    print_step(step_num, {"type": "investigate", "field": "age"}, result)
    step_num += 1
    
    result = env.step({"type": "investigate", "field": "biomarkers.EGFR"})
    print_step(step_num, {"type": "investigate", "field": "biomarkers.EGFR"}, result)
    step_num += 1
    
    # Check criteria for multiple trials
    for trial_id in obs['available_trials'][:3]:
        result = env.step({"type": "check_criteria", "trial_id": trial_id})
        print_step(step_num, {"type": "check_criteria", "trial_id": trial_id}, result)
        step_num += 1
    
    # Find an eligible trial by checking all
    eligible_trial = None
    for trial_id in obs['available_trials']:
        trial_result = env.step({"type": "check_criteria", "trial_id": trial_id})
        if trial_result['info'].get('inclusion_pass') and not trial_result['info'].get('exclusion_triggered'):
            eligible_trial = trial_id
            break
    
    if eligible_trial:
        result = env.step({"type": "select_trial", "trial_id": eligible_trial})
        print_step(step_num, {"type": "select_trial", "trial_id": eligible_trial}, result)
        step_num += 1
    else:
        # Select first trial as fallback
        result = env.step({"type": "select_trial", "trial_id": obs['available_trials'][0]})
        print_step(step_num, {"type": "select_trial", "trial_id": obs['available_trials'][0]}, result)
        step_num += 1
    
    # Resolve
    result = env.step({"type": "resolve"})
    print_step(step_num, {"type": "resolve"}, result)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT")
    print(f"{'='*80}")
    print(f"Total Reward: {result['total_reward']:.2f}")
    print(f"Outcome: {'✅ CORRECT' if result['info']['correct'] else '❌ WRONG'}")
    print(f"Decision Reward: {1.0 if result['info']['correct'] else -1.0:+.2f}")
    print(f"Efficiency Bonus: {'+0.2' if result['steps_taken'] <= 5 else '0.0'}")
    print(f"Analysis: {'Strong positive reward' if result['total_reward'] > 0.5 else 'Not strongly positive'}")


def example_wrong_decision():
    """Example 2: Wrong decision → clearly negative."""
    print_header("EXAMPLE 2: WRONG DECISION (Clearly Negative)")
    
    env = ClinicalTrialEnv(num_trials=5, max_steps=20)
    obs = env.reset(patient_seed=100, trial_seed=200)
    
    print(f"\n🔄 RESET")
    print(f"Available Trials: {obs['available_trials']}")
    
    step_num = 1
    
    # Check a few trials
    for trial_id in obs['available_trials'][:3]:
        result = env.step({"type": "check_criteria", "trial_id": trial_id})
        print_step(step_num, {"type": "check_criteria", "trial_id": trial_id}, result)
        step_num += 1
    
    # Deliberately select an ineligible trial
    ineligible_trial = None
    for trial_id in obs['available_trials']:
        trial_result = env.step({"type": "check_criteria", "trial_id": trial_id})
        if not trial_result['info'].get('inclusion_pass') or trial_result['info'].get('exclusion_triggered'):
            ineligible_trial = trial_id
            break
    
    if ineligible_trial:
        result = env.step({"type": "select_trial", "trial_id": ineligible_trial})
        print_step(step_num, {"type": "select_trial", "trial_id": ineligible_trial}, result)
        step_num += 1
    else:
        # Select last trial as fallback
        result = env.step({"type": "select_trial", "trial_id": obs['available_trials'][-1]})
        print_step(step_num, {"type": "select_trial", "trial_id": obs['available_trials'][-1]}, result)
        step_num += 1
    
    # Resolve
    result = env.step({"type": "resolve"})
    print_step(step_num, {"type": "resolve"}, result)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT")
    print(f"{'='*80}")
    print(f"Total Reward: {result['total_reward']:.2f}")
    print(f"Outcome: {'✅ CORRECT' if result['info']['correct'] else '❌ WRONG'}")
    print(f"Decision Reward: {1.0 if result['info']['correct'] else -1.0:+.2f}")
    print(f"Intermediate Rewards: {result['total_reward'] - (1.0 if result['info']['correct'] else -1.0):.2f}")
    print(f"Analysis: {'Clearly negative' if result['total_reward'] < 0 else 'NOT NEGATIVE (BUG!)'}")


def example_spam_actions():
    """Example 3: Spam actions → net negative (no reward hacking)."""
    print_header("EXAMPLE 3: SPAM ACTIONS (Reward Hacking Prevention)")
    
    env = ClinicalTrialEnv(num_trials=5, max_steps=20)
    obs = env.reset(patient_seed=150, trial_seed=250)
    
    print(f"\n🔄 RESET")
    print(f"Available Trials: {obs['available_trials']}")
    
    step_num = 1
    
    print(f"\n{'='*80}")
    print(f"ATTEMPTING REWARD HACKING: Spam check_criteria() for +0.05 each")
    print(f"{'='*80}")
    
    # Try to spam check_criteria for rewards
    for trial_id in obs['available_trials']:
        result = env.step({"type": "check_criteria", "trial_id": trial_id})
        print_step(step_num, {"type": "check_criteria", "trial_id": trial_id}, result)
        step_num += 1
    
    print(f"\nCurrent Total: {result['total_reward']:.2f}")
    print(f"Max possible from checking: {len(obs['available_trials']) * 0.05:.2f}")
    
    # Try to repeat checks (should get penalties)
    print(f"\n{'='*80}")
    print(f"ATTEMPTING: Repeat checks (should get penalties)")
    print(f"{'='*80}")
    
    for trial_id in obs['available_trials'][:2]:
        result = env.step({"type": "check_criteria", "trial_id": trial_id})
        print_step(step_num, {"type": "check_criteria", "trial_id": trial_id}, result)
        step_num += 1
    
    print(f"\nCurrent Total: {result['total_reward']:.2f}")
    
    # Spam investigate
    print(f"\n{'='*80}")
    print(f"ATTEMPTING: Spam investigate (should be neutral/negative)")
    print(f"{'='*80}")
    
    fields = ["age", "gender", "stage", "age", "gender"]  # Includes repeats
    for field in fields:
        result = env.step({"type": "investigate", "field": field})
        print_step(step_num, {"type": "investigate", "field": field}, result)
        step_num += 1
    
    print(f"\nCurrent Total: {result['total_reward']:.2f}")
    
    # Now make wrong decision
    result = env.step({"type": "select_trial", "trial_id": obs['available_trials'][-1]})
    print_step(step_num, {"type": "select_trial", "trial_id": obs['available_trials'][-1]}, result)
    step_num += 1
    
    result = env.step({"type": "resolve"})
    print_step(step_num, {"type": "resolve"}, result)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT")
    print(f"{'='*80}")
    print(f"Total Reward: {result['total_reward']:.2f}")
    print(f"Outcome: {'✅ CORRECT' if result['info']['correct'] else '❌ WRONG'}")
    print(f"Steps Taken: {result['steps_taken']}")
    print(f"\nREWARD BREAKDOWN:")
    print(f"  Max check_criteria rewards: {len(obs['available_trials']) * 0.05:.2f}")
    print(f"  Penalties from repeats: ~{-0.05 * 2:.2f}")
    print(f"  Final decision: {1.0 if result['info']['correct'] else -1.0:+.2f}")
    print(f"  Total: {result['total_reward']:.2f}")
    print(f"\nAnalysis: {'Reward hacking prevented - still negative!' if result['total_reward'] < 0 else 'REWARD HACKING POSSIBLE (BUG!)'}")


def example_efficient_correct():
    """Example 4: Efficient + correct → maximum reward."""
    print_header("EXAMPLE 4: EFFICIENT + CORRECT (Maximum Reward)")
    
    env = ClinicalTrialEnv(num_trials=5, max_steps=20)
    obs = env.reset(patient_seed=200, trial_seed=300)
    
    print(f"\n🔄 RESET")
    print(f"Available Trials: {obs['available_trials']}")
    
    step_num = 1
    
    # Minimal actions - check first trial
    result = env.step({"type": "check_criteria", "trial_id": obs['available_trials'][0]})
    print_step(step_num, {"type": "check_criteria", "trial_id": obs['available_trials'][0]}, result)
    step_num += 1
    
    # Check if eligible
    if result['info'].get('inclusion_pass') and not result['info'].get('exclusion_triggered'):
        eligible_trial = obs['available_trials'][0]
    else:
        # Check second trial
        result = env.step({"type": "check_criteria", "trial_id": obs['available_trials'][1]})
        print_step(step_num, {"type": "check_criteria", "trial_id": obs['available_trials'][1]}, result)
        step_num += 1
        eligible_trial = obs['available_trials'][1]
    
    # Select
    result = env.step({"type": "select_trial", "trial_id": eligible_trial})
    print_step(step_num, {"type": "select_trial", "trial_id": eligible_trial}, result)
    step_num += 1
    
    # Resolve
    result = env.step({"type": "resolve"})
    print_step(step_num, {"type": "resolve"}, result)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT")
    print(f"{'='*80}")
    print(f"Total Reward: {result['total_reward']:.2f}")
    print(f"Outcome: {'✅ CORRECT' if result['info']['correct'] else '❌ WRONG'}")
    print(f"Steps Taken: {result['steps_taken']}")
    print(f"\nREWARD BREAKDOWN:")
    print(f"  Decision: {1.0 if result['info']['correct'] else -1.0:+.2f}")
    print(f"  Efficiency bonus: {'+0.2' if result['steps_taken'] <= 5 else '0.0'}")
    print(f"  Check rewards: ~{0.05 * min(step_num - 2, 2):.2f}")
    print(f"  Total: {result['total_reward']:.2f}")
    print(f"\nAnalysis: {'Maximum reward achieved!' if result['total_reward'] >= 1.2 else 'Not maximum'}")


def main():
    """Run all examples."""
    example_correct_decision()
    example_wrong_decision()
    example_spam_actions()
    example_efficient_correct()
    
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "SUMMARY" + " "*46 + "║")
    print("╚" + "="*78 + "╝")
    print("\n✅ Decision dominance enforced:")
    print("   - Correct decision: ~1.0+ (strongly positive)")
    print("   - Wrong decision: <0 (clearly negative)")
    print("   - Spam actions: Cannot overcome -1.0 penalty")
    print("\n✅ Reward hacking prevented:")
    print("   - Max intermediate rewards: ~0.25 (5 trials × 0.05)")
    print("   - Cannot offset -1.0 wrong decision penalty")
    print("   - Agent must make correct decision to end positive")
    print("\n✅ Efficiency incentivized:")
    print("   - +0.2 bonus for ≤5 steps")
    print("   - Combined with +1.0 correct = 1.2+ maximum reward")
    print()


if __name__ == "__main__":
    main()
