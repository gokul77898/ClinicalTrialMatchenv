"""
Phase 3: Deterministic Clinical Trial Matching Agent

A structured, rule-based agent that follows a strict decision-making process
to select the best clinical trial for a patient.

This is NOT an LLM agent - it uses deterministic logic and clear reasoning steps.
"""

from typing import Dict, List, Set, Optional
from src.environment import ClinicalTrialEnv, Action, Observation


class ClinicalTrialAgent:
    """
    Deterministic agent for clinical trial matching.
    
    Follows a strict 5-step process:
    1. Investigate key fields (age, cancer_type, optionally PD_L1)
    2. Filter trials by cancer_type match
    3. Evaluate trials (check criteria, avoid exclusions)
    4. Select best candidate
    5. Resolve
    """
    
    def __init__(self):
        """Initialize agent with empty memory."""
        self.investigated_fields: Set[str] = set()
        self.checked_trials: Set[str] = set()
        self.selected_trial: Optional[str] = None
        self.episode_steps: int = 0
    
    def run_episode(self, env: ClinicalTrialEnv) -> Dict:
        """
        Run a complete episode following a simplified but effective strategy.
        
        Args:
            env: ClinicalTrial environment
            
        Returns:
            Dict with episode results
        """
        # Reset memory
        self.investigated_fields = set()
        self.checked_trials = set()
        self.selected_trial = None
        self.episode_steps = 0
        
        obs = env.reset()
        
        # STEP 1: Investigate cancer_type (most important)
        self.investigate(env, "cancer_type")
        
        # STEP 2: Get matching trials
        obs = env.state()
        matching_trials = []
        for trial in obs.available_trials:
            if trial["cancer_type"] == obs.patient.cancer_type:
                matching_trials.append(trial["trial_id"])
        
        # STEP 3: Check matching trials more thoroughly
        selected_trial = None
        if matching_trials:
            # Check up to 3 matching trials to find one without exclusions
            for trial_id in matching_trials[:3]:
                if self.episode_steps >= 6:  # Stop if approaching step limit
                    break
                
                result = self.check_trial(env, trial_id)
                
                if result:
                    info = result.get("info", {})
                    # Select if eligible and no exclusion triggered
                    if info.get("eligible", False) and not info.get("exclusion_triggered", False):
                        selected_trial = trial_id
                        break
            
            # If no eligible trial found, check non-matching trials as fallback
            if not selected_trial:
                for trial in obs.available_trials[:2]:
                    if trial["trial_id"] not in matching_trials and self.episode_steps < 6:
                        result = self.check_trial(env, trial["trial_id"])
                        if result and result.get("info", {}).get("eligible", False):
                            selected_trial = trial["trial_id"]
                            break
            
            # Last resort: select first available trial
            if not selected_trial:
                selected_trial = obs.available_trials[0]["trial_id"]
        else:
            # No matching trials, check first available
            selected_trial = obs.available_trials[0]["trial_id"]
        
        env.step(Action(type="select_trial", trial_id=selected_trial))
        self.selected_trial = selected_trial
        self.episode_steps += 1
        
        # STEP 4: Resolve
        final_obs, reward, done, info = env.step(Action(type="resolve"))
        self.episode_steps += 1
        
        return {
            "selected_trial": self.selected_trial,
            "reward": reward.value,
            "steps": self.episode_steps,
            "success": info.get("correct", False)
        }
    
    def investigate(self, env: ClinicalTrialEnv, field: str) -> None:
        """
        Investigate a specific patient field.
        
        Args:
            env: ClinicalTrial environment
            field: Field to investigate (e.g., "age", "cancer_type", "biomarkers.PD_L1")
        """
        if field not in self.investigated_fields:
            env.step(Action(type="investigate", field=field))
            self.investigated_fields.add(field)
            self.episode_steps += 1
    
    def check_trial(self, env: ClinicalTrialEnv, trial_id: str) -> Dict:
        """
        Check eligibility criteria for a specific trial.
        
        Args:
            env: ClinicalTrial environment
            trial_id: Trial ID to check
            
        Returns:
            Dict with check results
        """
        if trial_id not in self.checked_trials:
            obs, reward, done, info = env.step(Action(type="check_criteria", trial_id=trial_id))
            self.checked_trials.add(trial_id)
            self.episode_steps += 1
            
            return {
                "observation": obs,
                "reward": reward,
                "done": done,
                "info": info
            }
        
        return {}
    
    def select_best_trial(self, env: ClinicalTrialEnv, candidates: List[str], fallback: List[str]) -> str:
        """
        Select the best trial from candidates or fallback.
        
        Args:
            env: ClinicalTrial environment
            candidates: List of eligible trials
            fallback: List of filtered trials (fallback options)
            
        Returns:
            Selected trial ID
        """
        # Get current observation to see available trials
        obs = env.state()
        available_trial_ids = [trial["trial_id"] for trial in obs.available_trials]
        
        selected = None
        
        if candidates:
            # Select first candidate that's actually available
            for candidate in candidates:
                if candidate in available_trial_ids:
                    selected = candidate
                    break
        
        if not selected and fallback:
            # Select first filtered trial that's actually available
            for trial_id in fallback:
                if trial_id in available_trial_ids:
                    selected = trial_id
                    break
        
        if not selected and available_trial_ids:
            # Last resort: select first available trial
            selected = available_trial_ids[0]
        
        if selected:
            env.step(Action(type="select_trial", trial_id=selected))
            self.selected_trial = selected
            self.episode_steps += 1
        else:
            # No trial selected - this shouldn't happen but handle gracefully
            self.selected_trial = "none"
        
        return selected or "none"
    
    def _investigate_key_fields(self, env: ClinicalTrialEnv, obs: Observation) -> None:
        """
        STEP 1: Investigate key fields (MAX 2).
        
        Focus on:
        - cancer_type (for filtering)
        - age (only if needed)
        """
        # Always investigate cancer_type first (most important for filtering)
        self.investigate(env, "cancer_type")
        
        # Only investigate age if we have time and it's crucial
        if self.episode_steps < 3:
            self.investigate(env, "age")
    
    def _filter_trials_by_cancer_type(self, obs: Observation) -> List[str]:
        """
        STEP 2: Filter trials by cancer_type match.
        
        Args:
            obs: Current observation
            
        Returns:
            List of trial IDs with matching cancer_type
        """
        # Get patient cancer type from observation
        # In the actual environment, patient info is in obs.patient
        patient_cancer_type = getattr(obs, 'patient', None)
        if patient_cancer_type and hasattr(patient_cancer_type, 'cancer_type'):
            patient_cancer_type = patient_cancer_type.cancer_type
        else:
            # Fallback: return all trials if we can't determine cancer type
            return [trial["trial_id"] for trial in obs.available_trials]
        
        # Filter trials by cancer_type
        filtered = []
        for trial in obs.available_trials:
            if trial.get("cancer_type") == patient_cancer_type:
                filtered.append(trial["trial_id"])
        
        return filtered if filtered else [trial["trial_id"] for trial in obs.available_trials]
    
    def _evaluate_trials(self, env: ClinicalTrialEnv, trial_ids: List[str]) -> List[str]:
        """
        STEP 3: Evaluate trials (MAX 4 trials).
        
        For each trial:
        - Check criteria
        - Discard if exclusion triggered
        - Mark as candidate if inclusion passes
        
        Args:
            env: ClinicalTrial environment
            trial_ids: List of trial IDs to evaluate
            
        Returns:
            List of candidate trial IDs
        """
        candidates = []
        max_evaluations = min(4, len(trial_ids))  # MAX 4 trials (increased from 3)
        
        for i, trial_id in enumerate(trial_ids[:max_evaluations]):
            if self.episode_steps >= 6:  # Stop if we're approaching step limit
                break
            
            result = self.check_trial(env, trial_id)
            
            if result:
                info = result.get("info", {})
                
                # Check if exclusion was triggered
                if info.get("exclusion_triggered", False):
                    continue  # Discard this trial
                
                # Check if inclusion passed
                if info.get("inclusion_pass", False):
                    candidates.append(trial_id)
        
        return candidates
    
    def _select_trial(self, candidates: List[str], filtered: List[str], env: ClinicalTrialEnv) -> None:
        """
        STEP 4: Selection logic.
        
        If candidates exist: select first candidate
        Else: select first filtered trial (fallback)
        """
        self.select_best_trial(env, candidates, filtered)


def random_agent(env: ClinicalTrialEnv) -> Dict:
    """
    Random baseline agent for comparison.
    
    Randomly selects a trial and resolves immediately.
    
    Args:
        env: ClinicalTrial environment
        
    Returns:
        Dict with episode results
    """
    obs = env.reset()
    
    # Randomly select first available trial
    if obs.available_trials:
        random_trial = obs.available_trials[0]["trial_id"]
        env.step(Action(type="select_trial", trial_id=random_trial))
    
    # Resolve immediately
    final_obs, reward, done, info = env.step(Action(type="resolve"))
    
    return {
        "selected_trial": random_trial if obs.available_trials else "none",
        "reward": reward.value,
        "steps": 2,  # select + resolve
        "success": info.get("correct", False)
    }
