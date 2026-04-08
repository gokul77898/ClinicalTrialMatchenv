"""
Phase 3B: Deterministic grader functions for clinical trial matching tasks.

Graders score agent performance from 0.0 to 1.0 based on episode history.
All graders are deterministic - same input always produces same output.
"""

from dataclasses import dataclass
from typing import Optional

from src.config import CLAMP_SCORES_FOR_HACKATHON


@dataclass
class EpisodeHistory:
    """
    Complete history of an agent's episode on a task.
    
    Used by graders to score agent performance.
    """
    task_id: str
    patient_seed: int
    trial_seeds: list[int]
    actions_taken: list[dict]
    final_selected_trial_id: Optional[str]
    final_reward: float
    steps_taken: int
    done: bool
    correct_trial_id: str


def grade_single_match(episode: EpisodeHistory) -> float:
    """
    Grade agent performance on single_match task (easy difficulty).
    
    Scoring:
    - Correct selection: 0.6
    - Checked criteria: 0.2
    - Efficiency: 0.2
    
    Args:
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    # BUG FIX 1: No selection = 0.0
    if episode.final_selected_trial_id is None:
        return 0.0
    
    # Component 1: Correct Selection (0.0 to 0.6)
    if episode.final_selected_trial_id == episode.correct_trial_id:
        selection_score = 0.6
    else:
        selection_score = 0.0
    
    # Component 2: Checked criteria before selecting (0.0 to 0.2)
    check_actions = [a for a in episode.actions_taken if a.get("type") == "check_criteria"]
    if len(check_actions) >= 1:
        criteria_score = 0.2
    else:
        criteria_score = 0.0
    
    # Component 3: Efficiency (0.0 to 0.2)
    # BUG FIX 2: Efficiency bonus only if correct
    if episode.final_selected_trial_id == episode.correct_trial_id:
        if episode.steps_taken <= 3:
            efficiency_score = 0.2
        elif episode.steps_taken <= 5:
            efficiency_score = 0.1
        else:
            efficiency_score = 0.0
    else:
        efficiency_score = 0.0
    
    # Final score
    raw = selection_score + criteria_score + efficiency_score
    if CLAMP_SCORES_FOR_HACKATHON:
        # Hackathon requires scores in (0, 1) - not 0.0 or 1.0
        clamped = max(0.01, min(0.99, raw))
    else:
        # Internal testing allows [0, 1]
        clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


def grade_hidden_exclusion(episode: EpisodeHistory) -> float:
    """
    Grade agent performance on hidden_exclusion task (medium difficulty).
    
    Scoring:
    - Correct selection: 0.5
    - Trial coverage: 0.2
    - Investigation: 0.15
    - Efficiency: 0.15
    
    Args:
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    # BUG FIX 1: No selection = 0.0
    if episode.final_selected_trial_id is None:
        return 0.0
    
    # Component 1: Correct Selection (0.0 to 0.5)
    if episode.final_selected_trial_id == episode.correct_trial_id:
        selection_score = 0.5
    else:
        selection_score = 0.0
    
    # Component 2: Checked all 5 trials (0.0 to 0.2)
    checked = set(a.get("trial_id") for a in episode.actions_taken 
                  if a.get("type") == "check_criteria" and a.get("trial_id") is not None)
    coverage = len(checked) / 5
    coverage_score = round(0.2 * coverage, 4)
    
    # Component 3: Investigated patient fields (0.0 to 0.15)
    investigated = set(a.get("field") for a in episode.actions_taken 
                       if a.get("type") == "investigate" and a.get("field") is not None)
    if len(investigated) >= 3:
        investigation_score = 0.15
    elif len(investigated) >= 1:
        investigation_score = 0.05
    else:
        investigation_score = 0.0
    
    # Component 4: Efficiency (0.0 to 0.15)
    # BUG FIX 2: Efficiency bonus only if correct
    if episode.final_selected_trial_id == episode.correct_trial_id:
        if episode.steps_taken <= 5:
            efficiency_score = 0.15
        elif episode.steps_taken <= 8:
            efficiency_score = 0.08
        else:
            efficiency_score = 0.0
    else:
        efficiency_score = 0.0
    
    # Final score
    raw = selection_score + coverage_score + investigation_score + efficiency_score
    if CLAMP_SCORES_FOR_HACKATHON:
        clamped = max(0.01, min(0.99, raw))
    else:
        clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


def grade_ambiguous_match(episode: EpisodeHistory) -> float:
    """
    Grade agent performance on ambiguous_match task (hard difficulty).
    
    Scoring:
    - Correct selection: 0.4
    - Biomarker investigation: 0.2
    - Trial coverage: 0.2
    - Lab investigation: 0.1
    - Efficiency: 0.1
    
    Args:
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    # BUG FIX 1: No selection = 0.0
    if episode.final_selected_trial_id is None:
        return 0.0
    
    # Component 1: Correct Selection (0.0 to 0.4)
    if episode.final_selected_trial_id == episode.correct_trial_id:
        selection_score = 0.4
    else:
        selection_score = 0.0
    
    # Component 2: Biomarker investigation (0.0 to 0.2)
    investigated = set(a.get("field") for a in episode.actions_taken
                       if a.get("type") == "investigate" and a.get("field") is not None)
    biomarker_fields = {"biomarkers.EGFR", "biomarkers.ALK", "biomarkers.PD_L1"}
    biomarkers_checked = investigated & biomarker_fields
    biomarker_score = round(0.2 * (len(biomarkers_checked) / 3), 4)
    
    # Component 3: Trial coverage (0.0 to 0.2)
    checked = set(a.get("trial_id") for a in episode.actions_taken
                  if a.get("type") == "check_criteria" and a.get("trial_id") is not None)
    coverage = len(checked) / 7
    coverage_score = round(0.2 * coverage, 4)
    
    # Component 4: Lab value investigation (0.0 to 0.1)
    lab_fields = {"lab_values.hb", "lab_values.wbc", "lab_values.creatinine"}
    labs_checked = investigated & lab_fields
    if len(labs_checked) >= 2:
        lab_score = 0.1
    elif len(labs_checked) >= 1:
        lab_score = 0.05
    else:
        lab_score = 0.0
    
    # Component 5: Efficiency (0.0 to 0.1)
    # BUG FIX 2: Efficiency bonus only if correct
    if episode.final_selected_trial_id == episode.correct_trial_id:
        if episode.steps_taken <= 8:
            efficiency_score = 0.1
        elif episode.steps_taken <= 12:
            efficiency_score = 0.05
        else:
            efficiency_score = 0.0
    else:
        efficiency_score = 0.0
    
    # Final score
    raw = selection_score + biomarker_score + coverage_score + lab_score + efficiency_score
    if CLAMP_SCORES_FOR_HACKATHON:
        clamped = max(0.01, min(0.99, raw))
    else:
        clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


def grade_multi_patient(episode: EpisodeHistory) -> float:
    """
    Grade agent performance on multi_patient task (expert difficulty).
    
    Scoring:
    - Correct selection per case: 0.3 each x 3 = 0.9 max
    - Efficiency across all cases: 0.1 max
    
    Args:
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    if episode.final_selected_trial_id is None and not episode.actions_taken:
        return 0.0
    
    # Count select_trial actions mapped to cases via switch_case context
    # Use episode.actions_taken to reconstruct per-case selections
    case_selections = {}
    current_case = "case_1"
    for action in episode.actions_taken:
        if action.get("type") == "switch_case":
            current_case = action.get("case_id", current_case)
        elif action.get("type") == "select_trial":
            case_selections[current_case] = action.get("trial_id")
    
    # Check correctness per case
    correct_ids = episode.correct_trial_id  # Will be comma-separated or list
    if isinstance(correct_ids, str) and "," in correct_ids:
        expected = [x.strip() for x in correct_ids.split(",")]
    elif isinstance(correct_ids, list):
        expected = correct_ids
    else:
        expected = []
    
    total_correct = 0
    for i, case_id in enumerate([f"case_{j+1}" for j in range(3)]):
        if i < len(expected) and case_selections.get(case_id) == expected[i]:
            total_correct += 1
    
    # Score: 0.3 per correct case
    case_score = total_correct * 0.3
    
    # Efficiency bonus only if all correct
    if total_correct == 3:
        if episode.steps_taken <= 12:
            efficiency_score = 0.1
        elif episode.steps_taken <= 16:
            efficiency_score = 0.05
        else:
            efficiency_score = 0.0
    else:
        efficiency_score = 0.0
    
    raw = case_score + efficiency_score
    if CLAMP_SCORES_FOR_HACKATHON:
        clamped = max(0.01, min(0.99, raw))
    else:
        clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


def grade_competing_trials(episode: EpisodeHistory) -> float:
    """
    Grade agent performance on competing_trials task (expert difficulty).
    
    Scoring:
    - Correct best trial selection: 0.5
    - Checked multiple trials: 0.2
    - Investigation of scores: 0.15
    - Efficiency: 0.15
    
    Args:
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    if episode.final_selected_trial_id is None:
        return 0.0
    
    # Component 1: Correct Selection (best trial by score)
    if episode.final_selected_trial_id == episode.correct_trial_id:
        selection_score = 0.5
    else:
        selection_score = 0.0
    
    # Component 2: Checked multiple trials (0.0 to 0.2)
    checked = set(a.get("trial_id") for a in episode.actions_taken
                  if a.get("type") == "check_criteria" and a.get("trial_id") is not None)
    if len(checked) >= 3:
        coverage_score = 0.2
    elif len(checked) >= 2:
        coverage_score = 0.1
    else:
        coverage_score = 0.0
    
    # Component 3: Investigation (0.0 to 0.15)
    investigated = set(a.get("field") for a in episode.actions_taken
                       if a.get("type") == "investigate" and a.get("field") is not None)
    if len(investigated) >= 2:
        investigation_score = 0.15
    elif len(investigated) >= 1:
        investigation_score = 0.05
    else:
        investigation_score = 0.0
    
    # Component 4: Efficiency (0.0 to 0.15)
    if episode.final_selected_trial_id == episode.correct_trial_id:
        if episode.steps_taken <= 6:
            efficiency_score = 0.15
        elif episode.steps_taken <= 10:
            efficiency_score = 0.08
        else:
            efficiency_score = 0.0
    else:
        efficiency_score = 0.0
    
    raw = selection_score + coverage_score + investigation_score + efficiency_score
    if CLAMP_SCORES_FOR_HACKATHON:
        clamped = max(0.01, min(0.99, raw))
    else:
        clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


def grade_contradictory_info(episode: EpisodeHistory) -> float:
    """
    Grade agent performance on contradictory_info task (expert difficulty).
    
    Scoring:
    - Correct trial selection: 0.4
    - Flagged contradiction: 0.3
    - Investigation before flagging: 0.15
    - Efficiency: 0.15
    
    Args:
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    if episode.final_selected_trial_id is None:
        return 0.0
    
    # Component 1: Correct Selection (0.0 to 0.4)
    if episode.final_selected_trial_id == episode.correct_trial_id:
        selection_score = 0.4
    else:
        selection_score = 0.0
    
    # Component 2: Flagged contradiction (0.0 to 0.3)
    flags = [a for a in episode.actions_taken if a.get("type") == "flag_contradiction"]
    if len(flags) >= 1:
        flag_score = 0.3
    else:
        flag_score = 0.0
    
    # Component 3: Investigation before flagging (0.0 to 0.15)
    investigated = set(a.get("field") for a in episode.actions_taken
                       if a.get("type") == "investigate" and a.get("field") is not None)
    lab_fields = {"lab_values.hb", "lab_values.wbc", "lab_values.creatinine"}
    labs_checked = investigated & lab_fields
    if len(labs_checked) >= 2:
        investigation_score = 0.15
    elif len(labs_checked) >= 1:
        investigation_score = 0.05
    else:
        investigation_score = 0.0
    
    # Component 4: Efficiency (0.0 to 0.15)
    if episode.final_selected_trial_id == episode.correct_trial_id:
        if episode.steps_taken <= 8:
            efficiency_score = 0.15
        elif episode.steps_taken <= 12:
            efficiency_score = 0.08
        else:
            efficiency_score = 0.0
    else:
        efficiency_score = 0.0
    
    raw = selection_score + flag_score + investigation_score + efficiency_score
    if CLAMP_SCORES_FOR_HACKATHON:
        clamped = max(0.01, min(0.99, raw))
    else:
        clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


def grade_logical_inference(episode: EpisodeHistory) -> float:
    """
    Grade agent performance on logical_inference task (brutal/expert difficulty).
    
    Scoring:
    - Correct selection: 0.4
    - Investigated biomarker expression: 0.15
    - Investigated lab values (detected unknown): 0.15
    - Investigated interaction-relevant fields (age + wbc): 0.15
    - Checked conflict (investigate_conflict action): 0.15
    
    Args:
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    if episode.final_selected_trial_id is None:
        return 0.0
    
    actions = episode.actions_taken
    investigated = set(
        a.get("field", "") for a in actions
        if a["type"] == "investigate"
    )
    
    # Correct selection
    if episode.final_selected_trial_id == episode.correct_trial_id:
        selection_score = 0.4
    else:
        selection_score = 0.0
    
    # Process scores
    process_score = 0.0
    
    # Checked biomarker expression
    if "biomarkers.EGFR_expression" in investigated:
        process_score += 0.15
    
    # Investigated lab values (creatinine is unknown)
    if "lab_values.creatinine" in investigated:
        process_score += 0.15
    
    # Investigated interaction-relevant fields
    if "age" in investigated and ("lab_values.wbc" in investigated or "lab_values.creatinine" in investigated):
        process_score += 0.15
    
    # Used investigate_conflict action
    conflict_checked = any(
        a["type"] == "investigate_conflict"
        for a in actions
    )
    if conflict_checked:
        process_score += 0.15
    
    raw = selection_score + process_score
    if CLAMP_SCORES_FOR_HACKATHON:
        clamped = max(0.01, min(0.99, raw))
    else:
        clamped = max(0.0, min(1.0, raw))
    return round(clamped, 4)


def grade_task(task_id: str, episode: EpisodeHistory) -> float:
    """
    Route to appropriate grader based on task_id.
    
    Args:
        task_id: Task identifier
        episode: Episode history
    
    Returns:
        float: Score between 0.0 and 1.0
    
    Raises:
        ValueError: If task_id is unknown
    """
    if task_id == "single_match":
        return grade_single_match(episode)
    elif task_id == "hidden_exclusion":
        return grade_hidden_exclusion(episode)
    elif task_id == "ambiguous_match":
        return grade_ambiguous_match(episode)
    elif task_id == "multi_patient":
        return grade_multi_patient(episode)
    elif task_id == "competing_trials":
        return grade_competing_trials(episode)
    elif task_id == "contradictory_info":
        return grade_contradictory_info(episode)
    elif task_id == "logical_inference":
        return grade_logical_inference(episode)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
