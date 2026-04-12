"""
OpenEnv-compliant Clinical Trial Matching Environment.

Fully typed environment with Observation, Action, and Reward models.
"""

from typing import Optional
from src.models import Observation, Action, Reward, PatientCase
from src.schemas.patient_schema import Patient, generate_random_patient
from src.schemas.trial_schema import ClinicalTrial, generate_random_trial
from src.engine.eligibility_engine import is_eligible, get_nested_value, check_inclusion, check_exclusion, get_eligibility_details
from src.tasks import get_task, TaskDefinition
from src.graders import grade_task, EpisodeHistory, _clamp


class ClinicalTrialEnv:
    """
    OpenEnv-compliant environment for clinical trial matching.
    
    Reward System (Decision-Dominant):
    - investigate(): 0 reward (neutral)
    - check_criteria(): +0.05 first time only
    - select_trial(): 0 reward (deferred)
    - resolve(): +1.0 correct / -1.0 wrong (DOMINANT)
    - Efficiency bonus: +0.2 if steps <= 5
    - Penalties: -0.05 for invalid/repeated actions
    """
    
    def __init__(self, num_trials: int = 5, max_steps: int = 20):
        """
        Initialize environment.
        
        Args:
            num_trials: Number of trials to generate per episode
            max_steps: Maximum steps allowed per episode
        """
        self.num_trials = num_trials
        self.max_steps = max_steps
        
        # Private internal state
        self._patient: Optional[Patient] = None
        self._trials: list[ClinicalTrial] = []
        self._selected_trial_id: Optional[str] = None
        self._steps_taken: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._investigated_fields: list[str] = []
        self._checked_trials: list[str] = []
        self._patient_seed: Optional[int] = None
        self._trial_seed: Optional[int] = None
        self._current_task: Optional[TaskDefinition] = None
        self._actions_history: list[dict] = []
        
        # Multi-patient mode state
        self._mode: str = "single"
        self._cases: list[dict] = []
        self._active_case_id: Optional[str] = None
    
    def reset(self, patient_seed: Optional[int] = None, trial_seed: Optional[int] = None, task_id: Optional[str] = None) -> Observation:
        """
        Reset environment for new episode.
        
        Args:
            patient_seed: Seed for patient generation
            trial_seed: Seed for trial generation
            task_id: Optional task ID to load predefined task
        
        Returns:
            Observation: Initial state of the environment
        """
        # Handle task-based reset
        if task_id is not None:
            task = get_task(task_id)
            self._current_task = task
            patient_seed = task.patient_seed
            
            # Generate trials from task seeds
            self._trials = []
            for seed in task.trial_seeds:
                self._trials.append(generate_random_trial(seed=seed))
            
            if task.mode == "multi":
                self._mode = "multi"
                self._cases = []
                trial_summaries = self._build_trial_summaries()
                for i, pseed in enumerate(task.patient_seeds):
                    p = generate_random_patient(seed=pseed)
                    self._cases.append({
                        "case_id": f"case_{i+1}",
                        "patient": p,
                        "selected_trial_id": None,
                        "resolved": False,
                        "checked_trials": [],
                        "investigated_fields": [],
                    })
                self._active_case_id = "case_1"
                self._patient = self._cases[0]["patient"]
            else:
                self._mode = "single"
                self._cases = []
                self._active_case_id = None
                self._patient = generate_random_patient(seed=patient_seed)
        else:
            self._current_task = None
            self._mode = "single"
            self._cases = []
            self._active_case_id = None
            
            # Generate patient
            self._patient = generate_random_patient(seed=patient_seed)
            
            # Generate trials
            self._trials = []
            for i in range(self.num_trials):
                seed = (trial_seed + i) if trial_seed is not None else None
                self._trials.append(generate_random_trial(seed=seed))
        
        # Reset all internal state
        self._selected_trial_id = None
        self._steps_taken = 0
        self._done = False
        self._total_reward = 0.0
        self._investigated_fields = []
        self._checked_trials = []
        self._patient_seed = patient_seed
        self._trial_seed = trial_seed
        self._actions_history = []
        self._flagged_contradictions = []
        
        return self._build_observation()
    
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one action in the environment.
        
        Args:
            action: Action to execute
        
        Returns:
            tuple: (observation, reward, done, info)
        
        Raises:
            RuntimeError: If episode is done or environment not initialized
        """
        if self._patient is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        
        # Increment step counter
        self._steps_taken += 1
        
        # Route to appropriate action handler
        if action.type == "investigate":
            reward_value, reason, info = self._action_investigate(action.field)
            self._actions_history.append({"type": "investigate", "field": action.field})
        elif action.type == "check_criteria":
            reward_value, reason, info = self._action_check_criteria(action.trial_id)
            self._actions_history.append({"type": "check_criteria", "trial_id": action.trial_id})
        elif action.type == "select_trial":
            reward_value, reason, info = self._action_select_trial(action.trial_id)
            self._actions_history.append({"type": "select_trial", "trial_id": action.trial_id})
        elif action.type == "switch_case":
            reward_value, reason, info = self._action_switch_case(action.case_id)
            self._actions_history.append({"type": "switch_case", "case_id": action.case_id})
        elif action.type == "flag_contradiction":
            reward_value, reason, info = self._action_flag_contradiction(action.reason)
            self._actions_history.append({"type": "flag_contradiction", "reason": action.reason})
        elif action.type == "investigate_conflict":
            reward_value, reason, info = self._action_investigate_conflict(action.field)
            self._actions_history.append({"type": "investigate_conflict", "field": action.field})
        elif action.type == "resolve":
            reward_value, reason, is_terminal, info = self._action_resolve()
            self._actions_history.append({"type": "resolve"})
            self._done = True
        else:
            reward_value = -0.05
            reason = f"Unknown action type: {action.type}"
            info = {"penalty": True}
        
        # Update total reward
        self._total_reward += reward_value
        
        # Check if max steps reached
        if self._steps_taken >= self.max_steps and not self._done:
            self._done = True
            reward_value = -0.5
            reason = "Max steps reached"
            is_terminal = True
            info["max_steps_reached"] = True
        
        # Build reward object
        is_terminal = self._done
        reward = Reward(
            value=reward_value,
            reason=reason,
            is_terminal=is_terminal,
            cumulative=self._total_reward
        )
        
        # Build observation
        observation = self._build_observation()
        
        # Add action info
        info["action"] = action.type
        info["raw_value"] = reward_value
        
        return observation, reward, self._done, info
    
    def state(self) -> Observation:
        """
        Get current state without taking an action.
        
        Returns:
            Observation: Current state of the environment
        
        Raises:
            RuntimeError: If environment not initialized
        """
        if self._patient is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        return self._build_observation()
    
    def _build_trial_summaries(self) -> list[dict]:
        """Build trial summary dicts from self._trials with randomization.
        
        Guarantees the correct trial (if task-defined) appears in the first 4
        positions so that agents operating under a 4-trial action budget can
        always reach it. This preserves solvability under realistic constraints.
        """
        import random
        
        available_trials = []
        # Create list of trials first
        trials_list = list(self._trials)
        # Randomize order to remove bias
        random.shuffle(trials_list)
        
        # Ensure correct trial is in first 4 positions (solvability guarantee)
        if self._current_task and hasattr(self._current_task, 'correct_trial_id'):
            correct_id = self._current_task.correct_trial_id
            correct_idx = None
            for i, t in enumerate(trials_list):
                if t.trial_id == correct_id:
                    correct_idx = i
                    break
            if correct_idx is not None and correct_idx >= 4:
                # Swap with a random position in 0-3
                swap_pos = random.randint(0, 3)
                trials_list[correct_idx], trials_list[swap_pos] = (
                    trials_list[swap_pos], trials_list[correct_idx]
                )
        
        for trial in trials_list:
            has_biomarker_req = (
                trial.required_biomarkers.EGFR is not None or
                trial.required_biomarkers.ALK is not None or
                trial.required_biomarkers.PD_L1 is not None
            )
            has_expr_thresholds = (
                trial.required_biomarkers.EGFR_expression_min is not None or
                trial.required_biomarkers.ALK_expression_min is not None
            )
            has_interaction_rules = len(trial.interaction_exclusions) > 0
            available_trials.append({
                "trial_id": trial.trial_id,
                "cancer_type": trial.cancer_type,
                "num_inclusion_rules": len(trial.inclusion_criteria),
                "num_exclusion_rules": len(trial.exclusion_criteria),
                "has_biomarker_requirements": has_biomarker_req,
                "has_expression_thresholds": has_expr_thresholds,
                "has_interaction_rules": has_interaction_rules,
                "max_patients": trial.max_patients,
                "enrolled_patients": trial.enrolled_patients,
                "has_capacity": trial.has_capacity,
                "days_until_deadline": trial.days_until_deadline,
                "is_urgent": trial.is_urgent,
                "trial_score": trial.trial_score
            })
        return available_trials
    
    def _get_active_case(self) -> Optional[dict]:
        """Get the active case dict in multi mode."""
        if self._mode != "multi":
            return None
        for c in self._cases:
            if c["case_id"] == self._active_case_id:
                return c
        return None
    
    def _build_observation(self) -> Observation:
        """
        Build Observation from current internal state.
        
        Returns:
            Observation: Current state representation
        """
        available_trials = self._build_trial_summaries()
        
        if self._mode == "multi":
            cases = []
            for c in self._cases:
                cases.append(PatientCase(
                    case_id=c["case_id"],
                    patient=c["patient"],
                    available_trials=available_trials,
                    selected_trial_id=c["selected_trial_id"],
                    resolved=c["resolved"],
                    grade=c.get("grade"),
                ))
            return Observation(
                mode="multi",
                patient=self._patient,
                available_trials=available_trials,
                cases=cases,
                active_case_id=self._active_case_id,
                steps_taken=self._steps_taken,
                max_steps=self.max_steps,
                investigated_fields=self._investigated_fields.copy(),
                checked_trials=self._checked_trials.copy(),
                selected_trial_id=self._selected_trial_id,
                done=self._done,
                total_reward=self._total_reward
            )
        
        return Observation(
            patient=self._patient,
            available_trials=available_trials,
            steps_taken=self._steps_taken,
            max_steps=self.max_steps,
            investigated_fields=self._investigated_fields.copy(),
            checked_trials=self._checked_trials.copy(),
            selected_trial_id=self._selected_trial_id,
            done=self._done,
            total_reward=self._total_reward
        )
    
    def _action_investigate(self, field: str) -> tuple[float, str, dict]:
        """
        Execute investigate action.
        
        Args:
            field: Field path to investigate
        
        Returns:
            tuple: (reward, reason, info)
        """
        # In multi mode, use per-case investigated_fields
        if self._mode == "multi":
            case = self._get_active_case()
            if field in case["investigated_fields"]:
                return -0.05, "Field already investigated (unnecessary action)", {"penalty": True}
            try:
                value = get_nested_value(case["patient"], field)
                case["investigated_fields"].append(field)
                self._investigated_fields.append(field)
                return 0.0, f"Field '{field}' = {value}", {"value": value}
            except ValueError:
                return -0.05, f"Invalid field: {field}", {"penalty": True}
        
        if field in self._investigated_fields:
            return -0.05, "Field already investigated (unnecessary action)", {"penalty": True}
        
        try:
            value = get_nested_value(self._patient, field)
            self._investigated_fields.append(field)
            
            # Return "unknown" for None values
            if value is None:
                return 0.0, f"Field '{field}' = unknown (lab test not performed)", {"value": "unknown"}
            
            # Check if this field has a conflict (REALISTIC_MODE only)
            field_base = field.split(".")[-1] if "." in field else field
            if hasattr(self._patient, 'conflicting_fields') and field_base in self._patient.conflicting_fields:
                conflict = self._patient.conflicting_fields[field_base]
                return 0.0, (
                    f"Field '{field}' = {value} "
                    f"(NOTE: conflicting report suggests {conflict['notes_say']}, "
                    f"confidence: {conflict['confidence']} — verify before matching)"
                ), {"value": value, "has_conflict": True}
            
            return 0.0, f"Field '{field}' = {value}", {"value": value}
        except ValueError as e:
            return -0.05, f"Invalid field: {field}", {"penalty": True}
    
    def _action_check_criteria(self, trial_id: str) -> tuple[float, str, dict]:
        """
        Execute check_criteria action with task-aware guidance levels.
        
        Easy tasks: full details + summary
        Medium tasks: boolean flags + hint
        Hard/Expert tasks: boolean flags only (minimal)
        """
        if trial_id not in [t.trial_id for t in self._trials]:
            return -0.05, f"Unknown trial_id: {trial_id}", {"penalty": True}
        
        # In multi mode, use per-case checked_trials
        if self._mode == "multi":
            case = self._get_active_case()
            if trial_id in case["checked_trials"]:
                return -0.05, "Trial already checked (unnecessary action)", {"penalty": True}
            trial = next(t for t in self._trials if t.trial_id == trial_id)
            details = get_eligibility_details(case["patient"], trial)
            case["checked_trials"].append(trial_id)
            self._checked_trials.append(trial_id)
        else:
            if trial_id in self._checked_trials:
                return -0.05, "Trial already checked (unnecessary action)", {"penalty": True}
            trial = next(t for t in self._trials if t.trial_id == trial_id)
            details = get_eligibility_details(self._patient, trial)
            self._checked_trials.append(trial_id)
        
        # Determine guidance level from current task difficulty
        difficulty = "easy"
        if self._current_task is not None:
            difficulty = getattr(self._current_task, 'difficulty', 'easy')
        
        if difficulty == "easy":
            # Full guidance — easy task must be solvable
            info = {
                "inclusion_pass": details["inclusion_pass"],
                "exclusion_triggered": details["exclusion_triggered"],
                "biomarkers_pass": details["biomarkers_pass"],
                "prior_treatments_pass": details.get(
                    "prior_treatment_details", {}
                ).get("passed", True),
                "capacity_available": details.get(
                    "capacity_details", {}
                ).get("has_capacity", True),
                "eligible": details["eligible"],
                "summary": details["summary"],
                "inclusion_details": details["inclusion_details"],
                "exclusion_details": details["exclusion_details"]
            }
        elif difficulty == "medium":
            # Partial guidance — tells what failed but not why
            info = {
                "inclusion_pass": details["inclusion_pass"],
                "exclusion_triggered": details["exclusion_triggered"],
                "biomarkers_pass": details["biomarkers_pass"],
                "prior_treatments_pass": details.get(
                    "prior_treatment_details", {}
                ).get("passed", True),
                "capacity_available": details.get(
                    "capacity_details", {}
                ).get("has_capacity", True),
                "eligible": details["eligible"],
                "hint": "Check patient fields that correspond to failed criteria"
            }
        else:
            # Minimal guidance — hard/expert tasks
            info = {
                "inclusion_pass": details["inclusion_pass"],
                "exclusion_triggered": details["exclusion_triggered"],
                "biomarkers_pass": details["biomarkers_pass"],
                "prior_treatments_pass": details.get(
                    "prior_treatment_details", {}
                ).get("passed", True),
                "capacity_available": details.get(
                    "capacity_details", {}
                ).get("has_capacity", True),
                "eligible": details["eligible"]
            }
        
        return 0.05, f"Criteria checked for {trial_id}", info
    
    def _action_select_trial(self, trial_id: str) -> tuple[float, str, dict]:
        """
        Execute select_trial action.
        
        Args:
            trial_id: Trial ID to select
        
        Returns:
            tuple: (reward, reason, info)
        """
        # Validate trial_id
        trial_ids = [t.trial_id for t in self._trials]
        if trial_id not in trial_ids:
            return -0.05, f"Unknown trial_id: {trial_id}", {"penalty": True}
        
        if self._mode == "multi":
            case = self._get_active_case()
            case["selected_trial_id"] = trial_id
            self._selected_trial_id = trial_id
            return 0.0, f"Selected trial {trial_id} for {case['case_id']}", {}
        
        self._selected_trial_id = trial_id
        return 0.0, f"Selected trial: {trial_id}", {}
    
    def _action_switch_case(self, case_id: str) -> tuple[float, str, dict]:
        """
        Switch active patient case in multi-patient mode.
        
        Args:
            case_id: Case ID to switch to
        
        Returns:
            tuple: (reward, reason, info)
        """
        if self._mode != "multi":
            return -0.05, "switch_case only valid in multi-patient mode", {"penalty": True}
        
        case_ids = [c["case_id"] for c in self._cases]
        if case_id not in case_ids:
            return -0.05, f"Unknown case_id: {case_id}", {"penalty": True}
        
        self._active_case_id = case_id
        case = self._get_active_case()
        self._patient = case["patient"]
        return 0.0, f"Switched to {case_id}", {}
    
    def _action_flag_contradiction(self, reason: str) -> tuple[float, str, dict]:
        """
        Flag a contradiction in patient data.
        
        Args:
            reason: Description of the contradiction found
        
        Returns:
            tuple: (reward, reason, info)
        """
        if not hasattr(self, '_flagged_contradictions'):
            self._flagged_contradictions = []
        
        if reason in self._flagged_contradictions:
            return -0.05, "Contradiction already flagged", {"penalty": True}
        
        self._flagged_contradictions.append(reason)
        return 0.1, f"Contradiction flagged: {reason}", {"flagged": reason}
    
    def _action_investigate_conflict(self, field: str) -> tuple[float, str, dict]:
        """
        Investigate a conflicting field in patient data.
        Returns detailed conflict info if conflict exists.
        """
        # STRICT_MODE: No conflicting_fields attribute
        if not hasattr(self._patient, 'conflicting_fields') or not self._patient.conflicting_fields:
            return -0.05, "No conflicts found in patient data", {"penalty": True}
        if field not in self._patient.conflicting_fields:
            return -0.05, f"No conflict for field: {field}", {}
        conflict = self._patient.conflicting_fields[field]
        return 0.1, f"Conflict resolved for {field}", {
            "field": field,
            "reported_value": conflict["reported"],
            "alternative_value": conflict["notes_say"],
            "confidence": conflict["confidence"],
            "recommendation": "Use reported value unless confidence is low"
        }
    
    def _action_resolve(self) -> tuple[float, str, bool, dict]:
        """
        Execute resolve action (end episode).
        
        Returns:
            tuple: (reward, reason, is_terminal, info)
        """
        # Multi-patient resolve
        if self._mode == "multi":
            return self._action_resolve_multi()
        
        if self._selected_trial_id is None:
            return -1.0, "Resolved without selecting a trial", True, {"correct": False}
        
        # Find selected trial
        selected_trial = next(t for t in self._trials if t.trial_id == self._selected_trial_id)
        
        # Check eligibility
        is_correct = is_eligible(self._patient, selected_trial)
        
        if is_correct:
            base_reward = 1.0
            reason = "Correct trial selection!"
            correct = True
        else:
            base_reward = -1.0
            reason = "Wrong trial selection (CRITICAL ERROR - patient not eligible)"
            correct = False
        
        # Add efficiency bonus
        efficiency_bonus = 0.2 if self._steps_taken <= 5 else 0.0
        reward = base_reward + efficiency_bonus
        
        if efficiency_bonus > 0:
            reason += f" (Efficient: {self._steps_taken} steps)"
        
        info = {
            "correct": correct,
            "selected_trial": self._selected_trial_id,
            "efficiency_bonus": efficiency_bonus
        }
        
        # Add grading if task-based episode
        if self._current_task is not None:
            episode = EpisodeHistory(
                task_id=self._current_task.task_id,
                patient_seed=self._current_task.patient_seed,
                trial_seeds=self._current_task.trial_seeds,
                actions_taken=self._actions_history,
                final_selected_trial_id=self._selected_trial_id,
                final_reward=self._total_reward + reward,  # Include current reward
                steps_taken=self._steps_taken,
                done=True,
                correct_trial_id=self._current_task.correct_trial_id
            )
            raw_grade = grade_task(self._current_task.task_id, episode)
            grade = _clamp(raw_grade)
            assert 0.0 < grade < 1.0, f"INVALID SCORE: {grade}"
            info["grade"] = grade
            info["task_id"] = self._current_task.task_id
            info["correct_trial_id"] = self._current_task.correct_trial_id
        else:
            info["grade"] = None
        
        return reward, reason, True, info
    
    def _action_resolve_multi(self) -> tuple[float, str, bool, dict]:
        """
        Resolve multi-patient episode.
        All cases must have a selected trial. Computes per-case correctness
        and average grade.
        """
        # Check all cases have selections
        unresolved = [c for c in self._cases if c["selected_trial_id"] is None]
        if unresolved:
            ids = [c["case_id"] for c in unresolved]
            return -0.1, f"Not all cases resolved yet (missing: {ids})", True, {"correct": False}
        
        # Grade each case
        task = self._current_task
        case_results = []
        total_correct = 0
        for i, case in enumerate(self._cases):
            selected_id = case["selected_trial_id"]
            trial = next(t for t in self._trials if t.trial_id == selected_id)
            correct = is_eligible(case["patient"], trial)
            expected = task.correct_trial_ids[i] if task and i < len(task.correct_trial_ids) else None
            if expected:
                correct = correct and (selected_id == expected)
            if correct:
                total_correct += 1
            case["resolved"] = True
            case_results.append({
                "case_id": case["case_id"],
                "selected": selected_id,
                "correct": correct,
            })
        
        # Score: 0.3 per correct case + efficiency
        case_score = total_correct * 0.3
        efficiency = 0.1 if self._steps_taken <= 12 else (0.05 if self._steps_taken <= 16 else 0.0)
        grade = _clamp(case_score + efficiency)
        assert 0.0 < grade < 1.0, f"INVALID SCORE: {grade}"
        
        reward = (1.0 if total_correct == len(self._cases) else 0.0) + (0.2 if self._steps_taken <= 12 else 0.0)
        reason = f"Multi-patient resolve: {total_correct}/{len(self._cases)} correct"
        
        info = {
            "correct": total_correct == len(self._cases),
            "case_results": case_results,
            "grade": grade,
        }
        if task:
            info["task_id"] = task.task_id
        
        return reward, reason, True, info
