"""
OpenEnv-compliant typed models for ClinicalTrialMatchEnv.

Defines Observation, Action, and Reward models for type-safe environment interaction.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional, Any
from src.schemas.patient_schema import Patient


class PatientCase(BaseModel):
    """A single patient case in multi-patient mode."""
    model_config = ConfigDict(extra='forbid')

    case_id: str
    patient: Patient
    available_trials: list[dict[str, Any]]
    selected_trial_id: Optional[str] = None
    resolved: bool = False
    grade: Optional[float] = None


class Observation(BaseModel):
    """
    Observation returned by reset() and step().
    
    Represents the current state of the environment visible to the agent.
    """
    model_config = ConfigDict(extra='forbid')
    
    # Single patient mode (backward compatible)
    patient: Optional[Patient] = None
    available_trials: list[dict[str, Any]] = Field(default_factory=list)
    selected_trial_id: Optional[str] = None
    
    # Multi patient mode
    mode: Literal["single", "multi"] = "single"
    cases: list[PatientCase] = Field(default_factory=list)
    active_case_id: Optional[str] = None
    
    # Shared fields
    steps_taken: int = 0
    max_steps: int = 20
    investigated_fields: list[str] = Field(default_factory=list)
    checked_trials: list[str] = Field(default_factory=list)
    done: bool = False
    total_reward: float = 0.0


class Action(BaseModel):
    """
    Action submitted to step().
    
    Type-safe action with validation for required fields based on action type.
    """
    model_config = ConfigDict(extra='forbid')
    
    type: Literal["investigate", "check_criteria", "select_trial", "resolve", "switch_case", "flag_contradiction", "investigate_conflict"]
    field: Optional[str] = None
    trial_id: Optional[str] = None
    case_id: Optional[str] = None
    reason: Optional[str] = None
    
    @field_validator('field', 'trial_id', 'case_id', 'reason')
    @classmethod
    def validate_action_params(cls, v, info):
        """Validate that required parameters are present based on action type."""
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate action parameters after model initialization."""
        if self.type == "investigate":
            if self.field is None:
                raise ValueError("Action type 'investigate' requires 'field' parameter")
        elif self.type == "check_criteria":
            if self.trial_id is None:
                raise ValueError("Action type 'check_criteria' requires 'trial_id' parameter")
        elif self.type == "select_trial":
            if self.trial_id is None:
                raise ValueError("Action type 'select_trial' requires 'trial_id' parameter")
        elif self.type == "switch_case":
            if self.case_id is None:
                raise ValueError("Action type 'switch_case' requires 'case_id' parameter")
        elif self.type == "flag_contradiction":
            if self.reason is None:
                raise ValueError("Action type 'flag_contradiction' requires 'reason' parameter")
        elif self.type == "investigate_conflict":
            if self.field is None:
                raise ValueError("Action type 'investigate_conflict' requires 'field' parameter")
        elif self.type == "resolve":
            pass


class Reward(BaseModel):
    """
    Reward returned by step().
    
    Contains reward value, explanation, terminal status, and cumulative total.
    """
    model_config = ConfigDict(extra='forbid')
    
    value: float
    reason: str
    is_terminal: bool
    cumulative: float
