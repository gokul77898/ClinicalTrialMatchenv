"""
Comprehensive test suite for Phase 2: Action System + Reward Mechanism

Tests:
1. Full episode with correct flow
2. Wrong decision penalty dominance
3. Loop abuse prevention
4. Reward hacking attempt
5. Efficiency bonus verification
"""

import pytest
from src.environment import ClinicalTrialEnv
from src.models import Action


class TestPhase2ActionSystem:
    """Test suite for action system and reward mechanism."""
    
    def test_full_episode_correct_flow(self):
        """TEST 1: Full episode with correct decision flow."""
        print("\n" + "="*80)
        print("TEST 1: FULL EPISODE (CORRECT FLOW)")
        print("="*80)
        
        env = ClinicalTrialEnv(num_trials=5, max_steps=20)
        obs = env.reset(patient_seed=42, trial_seed=100)
        
        print(f"\n🔄 RESET")
        print(f"Available Trials: {len(obs.available_trials)}")
        
        rewards = []
        
        # Step 1: investigate
        obs, reward, done, info = env.step(Action(type="investigate", field="age"))
        rewards.append(reward.value)
        print(f"\nStep 1: investigate(age)")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        assert reward.value == 0.0, "investigate should give 0 reward"
        assert not done, "Episode should not be done"
        
        # Step 2: check_criteria on first trial
        trial_id = obs.available_trials[0]["trial_id"]
        obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=trial_id))
        rewards.append(reward.value)
        print(f"\nStep 2: check_criteria({trial_id})")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        print(f"  Inclusion: {info.get('inclusion_pass')}, Exclusion: {info.get('exclusion_triggered')}")
        assert reward.value == 0.05, "First check_criteria should give +0.05"
        
        # Find an eligible trial
        eligible_trial = None
        for trial in obs.available_trials:
            tid = trial["trial_id"]
            obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=tid))
            if info.get('inclusion_pass') and not info.get('exclusion_triggered'):
                eligible_trial = tid
                print(f"\nFound eligible trial: {tid}")
                break
        
        if not eligible_trial:
            eligible_trial = obs.available_trials[0]["trial_id"]
            print(f"\nNo eligible trial found, using first trial: {eligible_trial}")
        
        # Step 3: select_trial
        obs, reward, done, info = env.step(Action(type="select_trial", trial_id=eligible_trial))
        rewards.append(reward.value)
        print(f"\nStep 3: select_trial({eligible_trial})")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        assert reward.value == 0.0, "select_trial should give 0 reward"
        
        # Step 4: resolve
        obs, reward, done, info = env.step(Action(type="resolve"))
        rewards.append(reward.value)
        print(f"\nStep 4: resolve()")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        print(f"  Correct: {info['correct']}")
        print(f"  Done: {done}")
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULT")
        print(f"{'='*80}")
        print(f"Total Reward: {reward.cumulative:.2f}")
        print(f"Reward Progression: {[f'{r:+.2f}' for r in rewards]}")
        
        assert done == True, "Episode should be done after resolve"
        
        if info['correct']:
            assert reward.cumulative > 0.5, f"Correct decision should yield positive reward, got {reward.cumulative}"
            print("✅ PASSED: Correct flow with positive reward")
        else:
            print("⚠️  Trial was ineligible, but flow correct")
    
    def test_wrong_decision_penalty(self):
        """TEST 2: Wrong decision penalty dominates."""
        print("\n" + "="*80)
        print("TEST 2: WRONG DECISION PENALTY")
        print("="*80)
        
        env = ClinicalTrialEnv(num_trials=5, max_steps=20)
        obs = env.reset(patient_seed=100, trial_seed=200)
        
        print(f"\n🔄 RESET")
        print(f"Available Trials: {len(obs.available_trials)}")
        
        # Check first trial
        trial_a = obs.available_trials[0]["trial_id"]
        obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=trial_a))
        print(f"\nStep 1: check_criteria({trial_a})")
        print(f"  Inclusion: {info.get('inclusion_pass')}, Exclusion: {info.get('exclusion_triggered')}")
        
        # Find an ineligible trial
        ineligible_trial = None
        for trial in obs.available_trials:
            tid = trial["trial_id"]
            obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=tid))
            if not info.get('inclusion_pass') or info.get('exclusion_triggered'):
                ineligible_trial = tid
                print(f"\nFound ineligible trial: {tid}")
                break
        
        if not ineligible_trial:
            ineligible_trial = obs.available_trials[-1]["trial_id"]
            print(f"\nUsing last trial as ineligible: {ineligible_trial}")
        
        # Select wrong trial
        obs, reward, done, info = env.step(Action(type="select_trial", trial_id=ineligible_trial))
        print(f"\nStep 2: select_trial({ineligible_trial}) [WRONG]")
        print(f"  Total: {reward.cumulative:.2f}")
        
        # Resolve
        obs, reward, done, info = env.step(Action(type="resolve"))
        print(f"\nStep 3: resolve()")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        print(f"  Correct: {info['correct']}")
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULT")
        print(f"{'='*80}")
        print(f"Total Reward: {reward.cumulative:.2f}")
        
        if not info['correct']:
            assert reward.cumulative < 0, f"Wrong decision should yield negative reward, got {reward.cumulative}"
            assert reward.value <= -0.8, f"Wrong decision penalty should be dominant, got {reward.value}"
            print("✅ PASSED: Wrong decision yields negative reward")
        else:
            print("⚠️  Trial was actually eligible")
    
    def test_loop_abuse_prevention(self):
        """TEST 3: Loop abuse prevention (repeated actions penalized)."""
        print("\n" + "="*80)
        print("TEST 3: LOOP ABUSE PREVENTION")
        print("="*80)
        
        env = ClinicalTrialEnv(num_trials=5, max_steps=20)
        obs = env.reset(patient_seed=150, trial_seed=250)
        
        print(f"\n🔄 RESET")
        
        rewards = []
        
        # Spam investigate same field
        print(f"\nSpamming investigate('age') 10 times:")
        for i in range(10):
            obs, reward, done, info = env.step(Action(type="investigate", field="age"))
            rewards.append(reward.value)
            print(f"  Iteration {i+1}: Reward {reward.value:+.2f}, Total {reward.cumulative:.2f}")
        
        print(f"\n{'='*80}")
        print(f"ABUSE RESULT")
        print(f"{'='*80}")
        print(f"Total Reward after spam: {reward.cumulative:.2f}")
        print(f"Reward breakdown: {[f'{r:+.2f}' for r in rewards]}")
        
        # Should have penalties
        penalty_count = sum(1 for r in rewards if r < 0)
        print(f"Penalties applied: {penalty_count}/10")
        
        assert penalty_count >= 9, f"Expected 9 penalties for repeated actions, got {penalty_count}"
        assert reward.cumulative < 0, f"Spam should yield negative total, got {reward.cumulative}"
        
        # Now resolve with no selection
        obs, reward, done, info = env.step(Action(type="resolve"))
        print(f"\nResolve without selection:")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        
        assert reward.cumulative < 0, f"Loop abuse + no selection should be negative, got {reward.cumulative}"
        print("✅ PASSED: Loop abuse prevented with penalties")
    
    def test_reward_hacking_attempt(self):
        """TEST 4: Reward hacking attempt (farming intermediate rewards)."""
        print("\n" + "="*80)
        print("TEST 4: REWARD HACKING ATTEMPT")
        print("="*80)
        
        env = ClinicalTrialEnv(num_trials=5, max_steps=20)
        obs = env.reset(patient_seed=200, trial_seed=300)
        
        print(f"\n🔄 RESET")
        print(f"Available Trials: {len(obs.available_trials)}")
        
        # Try to farm check_criteria rewards
        print(f"\nFarming check_criteria rewards:")
        intermediate_total = 0
        for trial in obs.available_trials:
            trial_id = trial["trial_id"]
            obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=trial_id))
            intermediate_total += reward.value
            print(f"  check_criteria({trial_id}): +{reward.value:.2f}, Total: {reward.cumulative:.2f}")
        
        print(f"\nIntermediate rewards farmed: {intermediate_total:.2f}")
        print(f"Max possible: {len(obs.available_trials) * 0.05:.2f}")
        
        # Find and select ineligible trial
        ineligible_trial = None
        for trial in obs.available_trials:
            tid = trial["trial_id"]
            # Already checked, so this will get penalty
            obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=tid))
            if not info.get('inclusion_pass') or info.get('exclusion_triggered'):
                ineligible_trial = tid
                break
        
        if not ineligible_trial:
            ineligible_trial = obs.available_trials[-1]["trial_id"]
        
        obs, reward, done, info = env.step(Action(type="select_trial", trial_id=ineligible_trial))
        print(f"\nSelect wrong trial: {ineligible_trial}")
        print(f"  Total: {reward.cumulative:.2f}")
        
        obs, reward, done, info = env.step(Action(type="resolve"))
        print(f"\nResolve (wrong decision):")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        
        print(f"\n{'='*80}")
        print(f"HACKING ATTEMPT RESULT")
        print(f"{'='*80}")
        print(f"Intermediate farmed: {intermediate_total:.2f}")
        print(f"Final decision: {reward.value:.2f}")
        print(f"Total Reward: {reward.cumulative:.2f}")
        
        if not info['correct']:
            assert reward.cumulative < 0, f"Hacking attempt with wrong decision should be negative, got {reward.cumulative}"
            assert intermediate_total < abs(reward.value), f"Intermediate rewards should not overcome penalty"
            print("✅ PASSED: Reward hacking prevented")
        else:
            print("⚠️  Trial was actually eligible")
    
    def test_efficiency_bonus(self):
        """TEST 5: Efficiency bonus verification."""
        print("\n" + "="*80)
        print("TEST 5: EFFICIENCY BONUS")
        print("="*80)
        
        env = ClinicalTrialEnv(num_trials=5, max_steps=20)
        obs = env.reset(patient_seed=250, trial_seed=350)
        
        print(f"\n🔄 RESET")
        print(f"Available Trials: {len(obs.available_trials)}")
        
        # Minimal steps - check and select first eligible
        steps = 0
        eligible_trial = None
        
        for trial in obs.available_trials:
            trial_id = trial["trial_id"]
            obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=trial_id))
            steps += 1
            print(f"\nStep {steps}: check_criteria({trial_id})")
            print(f"  Inclusion: {info.get('inclusion_pass')}, Exclusion: {info.get('exclusion_triggered')}")
            
            if info.get('inclusion_pass') and not info.get('exclusion_triggered'):
                eligible_trial = trial_id
                print(f"  ✓ Found eligible trial!")
                break
        
        if not eligible_trial:
            eligible_trial = obs.available_trials[0]["trial_id"]
            print(f"\nNo eligible trial found, using first: {eligible_trial}")
        
        obs, reward, done, info = env.step(Action(type="select_trial", trial_id=eligible_trial))
        steps += 1
        print(f"\nStep {steps}: select_trial({eligible_trial})")
        
        obs, reward, done, info = env.step(Action(type="resolve"))
        steps += 1
        print(f"\nStep {steps}: resolve()")
        print(f"  Reward: {reward.value:+.2f}, Total: {reward.cumulative:.2f}")
        print(f"  Correct: {info['correct']}")
        print(f"  Steps taken: {obs.steps_taken}")
        
        print(f"\n{'='*80}")
        print(f"EFFICIENCY RESULT")
        print(f"{'='*80}")
        print(f"Steps taken: {obs.steps_taken}")
        print(f"Total Reward: {reward.cumulative:.2f}")
        print(f"Efficiency bonus applied: {info.get('efficiency_bonus', 0):.2f}")
        
        if obs.steps_taken <= 5:
            assert 'efficiency_bonus' in info, "Efficiency bonus should be in info"
            assert info['efficiency_bonus'] == 0.2, f"Efficiency bonus should be 0.2, got {info.get('efficiency_bonus')}"
            print("✅ PASSED: Efficiency bonus applied")
        else:
            print(f"⚠️  Steps exceeded 5 ({obs.steps_taken}), no efficiency bonus expected")
        
        if info['correct']:
            expected_min = 1.0  # Base correct reward
            if obs.steps_taken <= 5:
                expected_min += 0.2  # Efficiency bonus
            assert reward.cumulative >= expected_min - 0.1, f"Expected at least {expected_min}, got {reward.cumulative}"


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "PHASE 2 TEST SUITE" + " "*39 + "║")
    print("╚" + "="*78 + "╝")
    
    test_suite = TestPhase2ActionSystem()
    
    tests = [
        ("Full Episode (Correct Flow)", test_suite.test_full_episode_correct_flow),
        ("Wrong Decision Penalty", test_suite.test_wrong_decision_penalty),
        ("Loop Abuse Prevention", test_suite.test_loop_abuse_prevention),
        ("Reward Hacking Attempt", test_suite.test_reward_hacking_attempt),
        ("Efficiency Bonus", test_suite.test_efficiency_bonus),
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
