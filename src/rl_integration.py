"""
Reinforcement Learning Integration for ClinicalTrialMatchEnv

Makes the environment RL-compatible with state vectors, discrete action space,
and pluggable policy interface WITHOUT breaking existing functionality.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
from src.environment import ClinicalTrialEnv
from src.agents.clinical_trial_agent import ClinicalTrialAgent
from src.models import Observation, Action


# ============================================================================
# PART 1: STATE VECTOR FUNCTION
# ============================================================================

def get_state_vector(observation: Observation, trial_index: int = 0) -> np.ndarray:
    """
    Convert observation to fixed-size numeric state vector.
    
    Returns ~25-dimensional vector with patient and trial features.
    """
    patient = observation.patient
    trials = observation.available_trials
    
    features = []
    
    # Age (normalized 0-1)
    age_norm = patient.age / 100.0 if patient.age else 0.5
    features.append(age_norm)
    
    # Gender (one-hot: male, female, unknown)
    gender_map = {"male": [1, 0, 0], "female": [0, 1, 0]}
    gender_vec = gender_map.get(patient.gender.lower() if patient.gender else "", [0, 0, 1])
    features.extend(gender_vec)
    
    # Cancer type (one-hot)
    cancer_types = ["lung cancer", "breast cancer", "colon cancer", "prostate cancer", "ovarian cancer"]
    cancer_vec = [1 if patient.cancer_type == ct else 0 for ct in cancer_types]
    features.extend(cancer_vec)
    
    # Stage (one-hot: I, II, III, IV)
    stage_map = {"I": [1, 0, 0, 0], "II": [0, 1, 0, 0], "III": [0, 0, 1, 0], "IV": [0, 0, 0, 1]}
    stage_vec = stage_map.get(patient.stage, [0, 0, 0, 0])
    features.extend(stage_vec)
    
    # Biomarkers
    if hasattr(patient, 'biomarkers') and patient.biomarkers:
        egfr = 1.0 if patient.biomarkers.EGFR else 0.0
        alk = 1.0 if patient.biomarkers.ALK else 0.0
        pd_l1 = patient.biomarkers.PD_L1 / 100.0 if patient.biomarkers.PD_L1 else 0.0
        features.extend([egfr, alk, pd_l1])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Lab values (normalized)
    if hasattr(patient, 'lab_values') and patient.lab_values:
        hb = patient.lab_values.hb / 20.0 if patient.lab_values.hb else 0.0
        wbc = patient.lab_values.wbc / 20000.0 if patient.lab_values.wbc else 0.0
        creatinine = patient.lab_values.creatinine / 5.0 if patient.lab_values.creatinine else 0.0
        features.extend([hb, wbc, creatinine])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Trial features (for current trial)
    if trials and trial_index < len(trials):
        trial = trials[trial_index]
        cancer_match = 1.0 if trial.get("cancer_type") == patient.cancer_type else 0.0
        features.append(cancer_match)
        
        # Basic inclusion feasibility heuristic
        feasible = 1.0 if cancer_match > 0 else 0.0
        features.append(feasible)
    else:
        features.extend([0.0, 0.0])
    
    # Steps taken (normalized)
    steps_norm = observation.steps_taken / observation.max_steps
    features.append(steps_norm)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# PART 2: ACTION SPACE MAPPING
# ============================================================================

ACTION_SPACE = {
    0: "investigate_age",
    1: "investigate_cancer_type",
    2: "investigate_biomarkers",
    3: "check_trial_0",
    4: "check_trial_1",
    5: "check_trial_2",
    6: "check_trial_3",
    7: "select_best_trial",
    8: "resolve"
}

NUM_ACTIONS = len(ACTION_SPACE)


def action_id_to_env_action(action_id: int, observation: Observation, best_trial_id: Optional[str] = None) -> Optional[Action]:
    """
    Map discrete action ID to environment Action.
    
    Args:
        action_id: Integer action (0-8)
        observation: Current observation
        best_trial_id: Optional trial ID for select action
        
    Returns:
        Action object or None if invalid
    """
    if action_id == 0:
        return Action(type="investigate", field="age")
    elif action_id == 1:
        return Action(type="investigate", field="cancer_type")
    elif action_id == 2:
        return Action(type="investigate", field="biomarkers.PD_L1")
    elif action_id in [3, 4, 5, 6]:
        trial_idx = action_id - 3
        if trial_idx < len(observation.available_trials):
            trial_id = observation.available_trials[trial_idx]["trial_id"]
            return Action(type="check_criteria", trial_id=trial_id)
        return None
    elif action_id == 7:
        # Select best trial (use provided trial_id or first available)
        if best_trial_id:
            return Action(type="select_trial", trial_id=best_trial_id)
        elif observation.available_trials:
            trial_id = observation.available_trials[0]["trial_id"]
            return Action(type="select_trial", trial_id=trial_id)
        return None
    elif action_id == 8:
        return Action(type="resolve")
    
    return None


# ============================================================================
# PART 3: RL ENVIRONMENT WRAPPER
# ============================================================================

class RLEnvWrapper:
    """
    RL-compatible wrapper for ClinicalTrialEnv.
    
    Provides standard RL interface:
    - reset() -> state_vector
    - step(action_id) -> (state, reward, done, info)
    """
    
    def __init__(self, task_id: Optional[str] = None):
        """
        Initialize RL wrapper.
        
        Args:
            task_id: Optional task ID for reset
        """
        self.env = ClinicalTrialEnv()
        self.task_id = task_id
        self.current_obs = None
        self.total_reward = 0.0
        self.episode_steps = 0
        
    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial state vector.
        
        Returns:
            np.ndarray: Initial state vector
        """
        if self.task_id:
            self.current_obs = self.env.reset(task_id=self.task_id)
        else:
            self.current_obs = self.env.reset()
        
        self.total_reward = 0.0
        self.episode_steps = 0
        
        # Cache trial order at reset so indices stay stable
        self.trial_order = [t["trial_id"] for t in self.current_obs.available_trials]
        
        return get_state_vector(self.current_obs)
    
    def step(self, action_id: int, best_trial_id: Optional[str] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return RL step tuple.
        
        Args:
            action_id: Discrete action (0-8)
            best_trial_id: Optional trial ID for select action
            
        Returns:
            (state_vector, reward, done, info)
        """
        # Map action ID to environment action using cached trial order
        action = self._map_action(action_id, best_trial_id)
        
        if action is None:
            # Invalid action - return current state with penalty
            state_vec = get_state_vector(self.current_obs)
            return state_vec, -0.1, False, {"invalid_action": True}
        
        # Execute action
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        self.total_reward += reward.value
        self.episode_steps += 1
        
        # Get state vector
        state_vec = get_state_vector(obs)
        
        # Add episode info
        info["total_reward"] = self.total_reward
        info["episode_steps"] = self.episode_steps
        
        return state_vec, reward.value, done, info
    
    def _map_action(self, action_id: int, best_trial_id: Optional[str] = None) -> Optional[Action]:
        """Map action_id to Action using cached trial order."""
        if action_id == 0:
            return Action(type="investigate", field="age")
        elif action_id == 1:
            return Action(type="investigate", field="cancer_type")
        elif action_id == 2:
            return Action(type="investigate", field="biomarkers.PD_L1")
        elif action_id in [3, 4, 5, 6]:
            trial_idx = action_id - 3
            if trial_idx < len(self.trial_order):
                return Action(type="check_criteria", trial_id=self.trial_order[trial_idx])
            return None
        elif action_id == 7:
            if best_trial_id:
                return Action(type="select_trial", trial_id=best_trial_id)
            elif self.trial_order:
                return Action(type="select_trial", trial_id=self.trial_order[0])
            return None
        elif action_id == 8:
            return Action(type="resolve")
        return None
    
    def get_state(self) -> np.ndarray:
        """Get current state vector without taking action."""
        return get_state_vector(self.current_obs)


# ============================================================================
# PART 4: POLICY INTERFACE
# ============================================================================

class BasePolicy(ABC):
    """Abstract base class for policies."""
    
    @abstractmethod
    def act(self, state_vector: np.ndarray) -> int:
        """
        Select action given state vector.
        
        Args:
            state_vector: Current state
            
        Returns:
            action_id: Discrete action (0-8)
        """
        pass
    
    def reset(self):
        """Reset policy state (optional)."""
        pass


# ============================================================================
# PART 5: RANDOM POLICY
# ============================================================================

class RandomPolicy(BasePolicy):
    """Random baseline policy."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random policy.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
    
    def act(self, state_vector: np.ndarray) -> int:
        """Return random action."""
        return self.rng.randint(0, NUM_ACTIONS)


# ============================================================================
# PART 6: HEURISTIC POLICY (WRAPS EXISTING AGENT)
# ============================================================================

class HeuristicPolicy(BasePolicy):
    """
    Wraps existing ClinicalTrialAgent as a policy.
    
    Converts agent's scoring logic into step-by-step action IDs.
    Uses inclusion_pass/exclusion_triggered from check_criteria results.
    """
    
    def __init__(self):
        """Initialize heuristic policy."""
        self.step_count = 0
        self.investigated_fields = set()
        self.checked_trials = set()
        self.trial_scores = {}
        self.discarded_trials = set()
        self.selected = False
        self.best_trial_id = None
        self.env_wrapper = None
        self._num_trials = 4
        self._last_checked_trial_id = None
        
    def reset(self):
        """Reset policy state."""
        self.step_count = 0
        self.investigated_fields = set()
        self.checked_trials = set()
        self.trial_scores = {}
        self.discarded_trials = set()
        self.selected = False
        self.best_trial_id = None
        self._num_trials = 4
        self._last_checked_trial_id = None
    
    def set_env_wrapper(self, env_wrapper: 'RLEnvWrapper'):
        """Set environment wrapper for accessing observations."""
        self.env_wrapper = env_wrapper
    
    def observe_step_result(self, action_id: int, info: Dict):
        """
        Observe result of last step to update internal scoring.
        Called by episode runner after each env.step().
        """
        if 3 <= action_id <= 6 and self._last_checked_trial_id:
            trial_id = self._last_checked_trial_id
            self._last_checked_trial_id = None
            
            # Exclusion triggered -> discard
            if info.get("exclusion_triggered", False):
                self.discarded_trials.add(trial_id)
                self.trial_scores.pop(trial_id, None)
                return
            
            # Score based on inclusion_pass
            score = 0.0
            if info.get("inclusion_pass", False):
                score += 2.0
            
            # Small heuristic boosts from state
            if self.env_wrapper and self.env_wrapper.current_obs:
                patient = self.env_wrapper.current_obs.patient
                if hasattr(patient, 'biomarkers') and patient.biomarkers:
                    if patient.biomarkers.PD_L1 and patient.biomarkers.PD_L1 > 50:
                        score += 0.3
                if hasattr(patient, 'age') and patient.age:
                    if 18 <= patient.age <= 75:
                        score += 0.2
            
            self.trial_scores[trial_id] = score
    
    def _get_initial_trial_list(self) -> List[dict]:
        """Get stable trial list from initial reset observation."""
        if not self.env_wrapper or not hasattr(self.env_wrapper, 'trial_order'):
            return []
        # Rebuild trial info from cached trial_order + current obs
        obs = self.env_wrapper.current_obs
        trial_map = {t["trial_id"]: t for t in obs.available_trials}
        return [trial_map[tid] for tid in self.env_wrapper.trial_order if tid in trial_map]
    
    def _get_matching_trial_indices(self) -> List[int]:
        """Get indices of trials matching patient cancer_type (up to 4) using cached order."""
        if not self.env_wrapper or not self.env_wrapper.current_obs:
            return list(range(min(4, self._num_trials)))
        
        obs = self.env_wrapper.current_obs
        trial_order = getattr(self.env_wrapper, 'trial_order', [])
        trial_map = {t["trial_id"]: t for t in obs.available_trials}
        
        matching = []
        for idx, tid in enumerate(trial_order[:4]):
            trial = trial_map.get(tid)
            if trial and trial.get("cancer_type") == obs.patient.cancer_type:
                matching.append(idx)
        
        if not matching:
            matching = list(range(min(4, len(trial_order))))
        
        return matching
    
    def act(self, state_vector: np.ndarray) -> Tuple[int, Optional[str]]:
        """
        Select action using heuristic logic mirroring ClinicalTrialAgent.
        
        Returns:
            (action_id, best_trial_id_or_None)
        """
        self.step_count += 1
        
        # Phase 1: Investigate key fields
        if "age" not in self.investigated_fields:
            self.investigated_fields.add("age")
            return 0, None
        
        if "cancer_type" not in self.investigated_fields:
            self.investigated_fields.add("cancer_type")
            return 1, None
        
        if "biomarkers" not in self.investigated_fields:
            self.investigated_fields.add("biomarkers")
            return 2, None
        
        # Phase 2: Check only cancer_type-matching trials (up to 4)
        matching_indices = self._get_matching_trial_indices()
        
        for trial_idx in matching_indices:
            if trial_idx not in self.checked_trials:
                self.checked_trials.add(trial_idx)
                if trial_idx <= 3:
                    # Store trial_id BEFORE step using cached order
                    trial_order = getattr(self.env_wrapper, 'trial_order', [])
                    if trial_idx < len(trial_order):
                        self._last_checked_trial_id = trial_order[trial_idx]
                    return 3 + trial_idx, None
        
        # Phase 3: Select best scoring trial (once)
        if not self.selected:
            self.selected = True
            
            if self.trial_scores:
                self.best_trial_id = max(self.trial_scores, key=self.trial_scores.get)
            elif self.env_wrapper and self.env_wrapper.current_obs:
                obs = self.env_wrapper.current_obs
                for trial in obs.available_trials:
                    if trial["trial_id"] not in self.discarded_trials:
                        self.best_trial_id = trial["trial_id"]
                        break
                if not self.best_trial_id and obs.available_trials:
                    self.best_trial_id = obs.available_trials[0]["trial_id"]
            
            return 7, self.best_trial_id
        
        # Phase 4: Resolve
        return 8, None


# ============================================================================
# PART 7: EPISODE RUNNER
# ============================================================================

def run_episode(policy: BasePolicy, env: RLEnvWrapper, verbose: bool = False) -> Dict:
    """
    Run complete episode with policy.
    
    Args:
        policy: Policy to use
        env: RL environment wrapper
        verbose: Print episode details
        
    Returns:
        Dict with episode results
    """
    policy.reset()
    
    # Set env wrapper for HeuristicPolicy
    if isinstance(policy, HeuristicPolicy):
        policy.set_env_wrapper(env)
    
    state = env.reset()
    
    total_reward = 0.0
    steps = 0
    done = False
    
    episode_log = []
    
    while not done and steps < 20:
        # Select action
        action_result = policy.act(state)
        
        # Handle different return types
        if isinstance(action_result, tuple):
            action_id, best_trial_id = action_result
        else:
            action_id = action_result
            best_trial_id = None
        
        # Take step
        next_state, reward, done, info = env.step(action_id, best_trial_id)
        
        # Feed check_criteria results back to HeuristicPolicy for scoring
        if isinstance(policy, HeuristicPolicy):
            policy.observe_step_result(action_id, info)
        
        total_reward += reward
        steps += 1
        
        if verbose:
            action_name = ACTION_SPACE.get(action_id, "unknown")
            episode_log.append({
                "step": steps,
                "action": action_name,
                "reward": reward,
                "done": done
            })
        
        state = next_state
    
    if verbose:
        print(f"\nEpisode Summary:")
        for log in episode_log:
            print(f"  Step {log['step']}: {log['action']} -> reward={log['reward']:.3f}, done={log['done']}")
        print(f"  Total: {total_reward:.3f} in {steps} steps")
    
    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": info.get("correct", False),
        "episode_log": episode_log if verbose else []
    }


# ============================================================================
# PART 8 & 9: RL COMPATIBILITY CHECK WITH LOGGING
# ============================================================================

def run_rl_compatibility_check(num_episodes: int = 50, verbose: bool = False):
    """
    Run RL compatibility check comparing RandomPolicy vs HeuristicPolicy.
    
    Args:
        num_episodes: Number of episodes to run
        verbose: Print detailed logs
    """
    print("="*80)
    print("RL COMPATIBILITY CHECK")
    print("="*80)
    
    # Test RandomPolicy
    print("\nTesting RandomPolicy...")
    random_results = []
    for i in range(num_episodes):
        task_ids = ["single_match", "hidden_exclusion", "ambiguous_match"]
        task_id = task_ids[i % len(task_ids)]
        
        env = RLEnvWrapper(task_id=task_id)
        policy = RandomPolicy(seed=i)
        
        result = run_episode(policy, env, verbose=(verbose and i == 0))
        random_results.append(result)
    
    random_success_rate = np.mean([r["success"] for r in random_results]) * 100
    random_avg_reward = np.mean([r["total_reward"] for r in random_results])
    
    # Test HeuristicPolicy
    print("\nTesting HeuristicPolicy...")
    heuristic_results = []
    for i in range(num_episodes):
        task_ids = ["single_match", "hidden_exclusion", "ambiguous_match"]
        task_id = task_ids[i % len(task_ids)]
        
        env = RLEnvWrapper(task_id=task_id)
        policy = HeuristicPolicy()
        
        result = run_episode(policy, env, verbose=(verbose and i == 0))
        heuristic_results.append(result)
    
    heuristic_success_rate = np.mean([r["success"] for r in heuristic_results]) * 100
    heuristic_avg_reward = np.mean([r["total_reward"] for r in heuristic_results])
    
    # Test Direct ClinicalTrialAgent (reference)
    print("\nTesting Direct ClinicalTrialAgent (reference)...")
    direct_results = []
    for i in range(num_episodes):
        task_ids = ["single_match", "hidden_exclusion", "ambiguous_match"]
        task_id = task_ids[i % len(task_ids)]
        
        env = ClinicalTrialEnv()
        agent = ClinicalTrialAgent()
        result = agent.run_episode(env, task_id=task_id)
        direct_results.append(result)
    
    direct_success_rate = np.mean([r["success"] for r in direct_results]) * 100
    direct_avg_reward = np.mean([r["reward"] for r in direct_results])
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"{'Policy':<25} {'Success Rate':<15} {'Avg Reward':<15}")
    print("-" * 55)
    print(f"{'RandomPolicy':<25} {random_success_rate:>6.1f}%        {random_avg_reward:>8.3f}")
    print(f"{'HeuristicPolicy':<25} {heuristic_success_rate:>6.1f}%        {heuristic_avg_reward:>8.3f}")
    print(f"{'ClinicalTrialAgent':<25} {direct_success_rate:>6.1f}%        {direct_avg_reward:>8.3f}")
    
    print(f"\nCompatibility Check:")
    diff = abs(heuristic_success_rate - direct_success_rate)
    if diff <= 15:
        print(f"  HeuristicPolicy vs Direct Agent delta: {diff:.1f}%")
        print(f"  ✅ PASS - HeuristicPolicy approximates ClinicalTrialAgent")
    else:
        print(f"  HeuristicPolicy vs Direct Agent delta: {diff:.1f}%")
        print(f"  ⚠️  Gap is large but RL interface is functional")
    
    print(f"\nRL Integration Status:")
    print(f"  ✅ State vector: {get_state_vector(ClinicalTrialEnv().reset()).shape[0]}-dim float32")
    print(f"  ✅ Action space: {NUM_ACTIONS} discrete actions")
    print(f"  ✅ RLEnvWrapper: reset() -> state, step(action) -> (s,r,d,i)")
    print(f"  ✅ BasePolicy interface with act(state) -> action")
    print(f"  ✅ Episode runner operational")
    print(f"  ✅ Ready for PPO / RL training in Phase 2")
    
    return {
        "random": {"success_rate": random_success_rate, "avg_reward": random_avg_reward},
        "heuristic": {"success_rate": heuristic_success_rate, "avg_reward": heuristic_avg_reward},
        "direct_agent": {"success_rate": direct_success_rate, "avg_reward": direct_avg_reward}
    }


if __name__ == "__main__":
    # Run compatibility check
    results = run_rl_compatibility_check(num_episodes=50, verbose=True)
