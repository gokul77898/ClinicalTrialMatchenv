"""
OpenEnv Compliance Test Suite

Tests that ClinicalTrialEnv conforms to OpenEnv specification with typed models.
"""

import pytest
from src.environment import ClinicalTrialEnv
from src.models import Observation, Action, Reward


class TestOpenEnvCompliance:
    """Test suite for OpenEnv compliance."""
    
    def test_reset_returns_observation(self):
        """Test 1: reset() returns Observation."""
        print("\n" + "="*80)
        print("TEST 1: reset() returns Observation")
        print("="*80)
        
        env = ClinicalTrialEnv()
        obs = env.reset(patient_seed=42, trial_seed=100)
        
        print(f"Type: {type(obs)}")
        print(f"Steps taken: {obs.steps_taken}")
        print(f"Done: {obs.done}")
        print(f"Total reward: {obs.total_reward}")
        print(f"Investigated fields: {obs.investigated_fields}")
        print(f"Checked trials: {obs.checked_trials}")
        print(f"Selected trial: {obs.selected_trial_id}")
        print(f"Available trials: {len(obs.available_trials)}")
        
        assert isinstance(obs, Observation), f"Expected Observation, got {type(obs)}"
        assert obs.steps_taken == 0, f"Expected steps_taken=0, got {obs.steps_taken}"
        assert obs.done == False, f"Expected done=False, got {obs.done}"
        assert obs.total_reward == 0.0, f"Expected total_reward=0.0, got {obs.total_reward}"
        assert obs.investigated_fields == [], f"Expected empty investigated_fields, got {obs.investigated_fields}"
        assert obs.checked_trials == [], f"Expected empty checked_trials, got {obs.checked_trials}"
        assert obs.selected_trial_id is None, f"Expected selected_trial_id=None, got {obs.selected_trial_id}"
        assert len(obs.available_trials) == 5, f"Expected 5 trials, got {len(obs.available_trials)}"
        
        print("✅ PASSED")
    
    def test_step_returns_correct_tuple_format(self):
        """Test 2: step() returns correct tuple format."""
        print("\n" + "="*80)
        print("TEST 2: step() returns correct tuple format")
        print("="*80)
        
        env = ClinicalTrialEnv()
        env.reset(patient_seed=42, trial_seed=100)
        action = Action(type="investigate", field="age")
        result = env.step(action)
        
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 4, f"Expected 4 elements, got {len(result)}"
        
        obs, reward, done, info = result
        
        print(f"Observation type: {type(obs)}")
        print(f"Reward type: {type(reward)}")
        print(f"Done type: {type(done)}")
        print(f"Info type: {type(info)}")
        
        assert isinstance(obs, Observation), f"Expected Observation, got {type(obs)}"
        assert isinstance(reward, Reward), f"Expected Reward, got {type(reward)}"
        assert isinstance(done, bool), f"Expected bool, got {type(done)}"
        assert isinstance(info, dict), f"Expected dict, got {type(info)}"
        
        print("✅ PASSED")
    
    def test_state_returns_observation(self):
        """Test 3: state() returns Observation."""
        print("\n" + "="*80)
        print("TEST 3: state() returns Observation")
        print("="*80)
        
        env = ClinicalTrialEnv()
        env.reset(patient_seed=42, trial_seed=100)
        state = env.state()
        
        print(f"State type: {type(state)}")
        
        assert isinstance(state, Observation), f"Expected Observation, got {type(state)}"
        
        print("✅ PASSED")
    
    def test_state_raises_if_not_initialized(self):
        """Test 4: state() raises if not initialized."""
        print("\n" + "="*80)
        print("TEST 4: state() raises if not initialized")
        print("="*80)
        
        env = ClinicalTrialEnv()
        
        with pytest.raises(RuntimeError) as exc_info:
            env.state()
        
        print(f"Error message: {str(exc_info.value)}")
        assert "not initialized" in str(exc_info.value).lower()
        
        print("✅ PASSED")
    
    def test_step_raises_if_episode_done(self):
        """Test 5: step() raises if episode done."""
        print("\n" + "="*80)
        print("TEST 5: step() raises if episode done")
        print("="*80)
        
        env = ClinicalTrialEnv()
        env.reset(patient_seed=42, trial_seed=100)
        trial_id = env.state().available_trials[0]["trial_id"]
        
        env.step(Action(type="select_trial", trial_id=trial_id))
        env.step(Action(type="resolve"))
        
        print("Episode ended, attempting another step...")
        
        with pytest.raises(RuntimeError) as exc_info:
            env.step(Action(type="investigate", field="age"))
        
        print(f"Error message: {str(exc_info.value)}")
        assert "done" in str(exc_info.value).lower()
        
        print("✅ PASSED")
    
    def test_action_model_validation(self):
        """Test 6: Action model validation."""
        print("\n" + "="*80)
        print("TEST 6: Action model validation")
        print("="*80)
        
        # investigate without field must fail
        print("Testing: investigate without field")
        with pytest.raises(Exception) as exc_info:
            Action(type="investigate")
        print(f"  ✓ Raised: {type(exc_info.value).__name__}")
        
        # check_criteria without trial_id must fail
        print("Testing: check_criteria without trial_id")
        with pytest.raises(Exception) as exc_info:
            Action(type="check_criteria")
        print(f"  ✓ Raised: {type(exc_info.value).__name__}")
        
        # valid investigate must pass
        print("Testing: valid investigate")
        action = Action(type="investigate", field="age")
        assert action.field == "age"
        print(f"  ✓ Created: {action.type} with field={action.field}")
        
        print("✅ PASSED")
    
    def test_reward_model_is_terminal(self):
        """Test 7: Reward model is_terminal."""
        print("\n" + "="*80)
        print("TEST 7: Reward model is_terminal")
        print("="*80)
        
        env = ClinicalTrialEnv()
        env.reset(patient_seed=42, trial_seed=100)
        obs, reward, done, info = env.step(Action(type="investigate", field="age"))
        
        print(f"After investigate:")
        print(f"  reward.is_terminal: {reward.is_terminal}")
        print(f"  done: {done}")
        
        assert reward.is_terminal == False, f"Expected is_terminal=False, got {reward.is_terminal}"
        assert done == False, f"Expected done=False, got {done}"
        
        print("✅ PASSED")
    
    def test_full_episode_reward_flow(self):
        """Test 8: Full episode reward flow."""
        print("\n" + "="*80)
        print("TEST 8: Full episode reward flow")
        print("="*80)
        
        env = ClinicalTrialEnv()
        obs = env.reset(patient_seed=42, trial_seed=100)
        trial_id = obs.available_trials[0]["trial_id"]
        
        print(f"Trial ID: {trial_id}")
        
        _, r1, _, _ = env.step(Action(type="investigate", field="age"))
        print(f"Step 1 (investigate): value={r1.value:+.2f}, cumulative={r1.cumulative:.2f}")
        
        _, r2, _, _ = env.step(Action(type="check_criteria", trial_id=trial_id))
        print(f"Step 2 (check_criteria): value={r2.value:+.2f}, cumulative={r2.cumulative:.2f}")
        
        _, r3, _, _ = env.step(Action(type="select_trial", trial_id=trial_id))
        print(f"Step 3 (select_trial): value={r3.value:+.2f}, cumulative={r3.cumulative:.2f}")
        
        obs_final, r4, done, info = env.step(Action(type="resolve"))
        print(f"Step 4 (resolve): value={r4.value:+.2f}, cumulative={r4.cumulative:.2f}")
        print(f"  done: {done}")
        print(f"  obs_final.done: {obs_final.done}")
        print(f"  r4.is_terminal: {r4.is_terminal}")
        
        assert done == True, f"Expected done=True, got {done}"
        assert obs_final.done == True, f"Expected obs_final.done=True, got {obs_final.done}"
        assert r4.is_terminal == True, f"Expected r4.is_terminal=True, got {r4.is_terminal}"
        
        expected_cumulative = r1.value + r2.value + r3.value + r4.value
        print(f"\nExpected cumulative: {expected_cumulative:.2f}")
        print(f"Actual cumulative: {r4.cumulative:.2f}")
        
        assert abs(r4.cumulative - expected_cumulative) < 0.01, \
            f"Expected cumulative={expected_cumulative:.2f}, got {r4.cumulative:.2f}"
        
        print("✅ PASSED")
    
    def test_reward_dominance(self):
        """Test 9: Reward dominance (correct > wrong)."""
        print("\n" + "="*80)
        print("TEST 9: Reward dominance (correct > wrong)")
        print("="*80)
        
        # Episode A: Try to find eligible trial
        print("\nEpisode A: Attempting to find eligible trial")
        env_a = ClinicalTrialEnv()
        obs_a = env_a.reset(patient_seed=42, trial_seed=100)
        
        eligible_trial = None
        for trial_summary in obs_a.available_trials:
            trial_id = trial_summary["trial_id"]
            _, r, _, info = env_a.step(Action(type="check_criteria", trial_id=trial_id))
            print(f"  Checked {trial_id}: inclusion={info.get('inclusion_pass')}, exclusion={info.get('exclusion_triggered')}")
            
            if info.get('inclusion_pass') and not info.get('exclusion_triggered'):
                eligible_trial = trial_id
                print(f"  ✓ Found eligible: {trial_id}")
                break
        
        if eligible_trial:
            env_a.step(Action(type="select_trial", trial_id=eligible_trial))
            _, r_a, _, info_a = env_a.step(Action(type="resolve"))
            print(f"Episode A result: reward={r_a.cumulative:.2f}, correct={info_a.get('correct')}")
        else:
            print("  No eligible trial found, using first trial")
            trial_id = obs_a.available_trials[0]["trial_id"]
            env_a.step(Action(type="select_trial", trial_id=trial_id))
            _, r_a, _, info_a = env_a.step(Action(type="resolve"))
            print(f"Episode A result: reward={r_a.cumulative:.2f}, correct={info_a.get('correct')}")
        
        # Episode B: Select blindly
        print("\nEpisode B: Select blindly")
        env_b = ClinicalTrialEnv()
        obs_b = env_b.reset(patient_seed=100, trial_seed=200)
        
        # Find ineligible trial
        ineligible_trial = None
        for trial_summary in obs_b.available_trials:
            trial_id = trial_summary["trial_id"]
            _, r, _, info = env_b.step(Action(type="check_criteria", trial_id=trial_id))
            
            if not info.get('inclusion_pass') or info.get('exclusion_triggered'):
                ineligible_trial = trial_id
                print(f"  Found ineligible: {trial_id}")
                break
        
        if ineligible_trial:
            env_b.step(Action(type="select_trial", trial_id=ineligible_trial))
        else:
            env_b.step(Action(type="select_trial", trial_id=obs_b.available_trials[-1]["trial_id"]))
        
        _, r_b, _, info_b = env_b.step(Action(type="resolve"))
        print(f"Episode B result: reward={r_b.cumulative:.2f}, correct={info_b.get('correct')}")
        
        print(f"\nComparison:")
        print(f"  Episode A (attempted correct): {r_a.cumulative:.2f}")
        print(f"  Episode B (wrong): {r_b.cumulative:.2f}")
        
        if info_a.get('correct') and not info_b.get('correct'):
            assert r_a.cumulative > r_b.cumulative, \
                f"Correct episode should have higher reward: {r_a.cumulative:.2f} vs {r_b.cumulative:.2f}"
            print("✅ PASSED: Correct > Wrong")
        else:
            print("⚠️  Skipped: Could not create correct vs wrong comparison")
    
    def test_observation_consistency(self):
        """Test 10: Observation consistency (state() matches after step())."""
        print("\n" + "="*80)
        print("TEST 10: Observation consistency")
        print("="*80)
        
        env = ClinicalTrialEnv()
        env.reset(patient_seed=42, trial_seed=100)
        
        obs_from_step, _, _, _ = env.step(Action(type="investigate", field="age"))
        obs_from_state = env.state()
        
        print(f"obs_from_step.steps_taken: {obs_from_step.steps_taken}")
        print(f"obs_from_state.steps_taken: {obs_from_state.steps_taken}")
        print(f"obs_from_step.investigated_fields: {obs_from_step.investigated_fields}")
        print(f"obs_from_state.investigated_fields: {obs_from_state.investigated_fields}")
        print(f"obs_from_step.total_reward: {obs_from_step.total_reward}")
        print(f"obs_from_state.total_reward: {obs_from_state.total_reward}")
        
        assert obs_from_step.steps_taken == obs_from_state.steps_taken, \
            f"steps_taken mismatch: {obs_from_step.steps_taken} vs {obs_from_state.steps_taken}"
        assert obs_from_step.investigated_fields == obs_from_state.investigated_fields, \
            f"investigated_fields mismatch"
        assert obs_from_step.total_reward == obs_from_state.total_reward, \
            f"total_reward mismatch: {obs_from_step.total_reward} vs {obs_from_state.total_reward}"
        
        print("✅ PASSED")


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "OPENENV COMPLIANCE TEST SUITE" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    
    test_suite = TestOpenEnvCompliance()
    
    tests = [
        ("reset() returns Observation", test_suite.test_reset_returns_observation),
        ("step() returns correct tuple format", test_suite.test_step_returns_correct_tuple_format),
        ("state() returns Observation", test_suite.test_state_returns_observation),
        ("state() raises if not initialized", test_suite.test_state_raises_if_not_initialized),
        ("step() raises if episode done", test_suite.test_step_raises_if_episode_done),
        ("Action model validation", test_suite.test_action_model_validation),
        ("Reward model is_terminal", test_suite.test_reward_model_is_terminal),
        ("Full episode reward flow", test_suite.test_full_episode_reward_flow),
        ("Reward dominance", test_suite.test_reward_dominance),
        ("Observation consistency", test_suite.test_observation_consistency),
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
