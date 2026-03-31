"""
Strict Clinical Trial Schema for Programmatic Eligibility Evaluation
Machine-readable trial definitions with NO free-text criteria.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError
from typing import Literal, Optional, Union, Any, ClassVar
import random


class Rule(BaseModel):
    """
    Eligibility rule for programmatic evaluation.
    
    Supports nested field access (e.g., "lab_values.hb").
    Operator must be valid comparison operator.
    Value type must match field semantics.
    
    Field-type mapping enforced:
    - age: int/float (numeric)
    - gender: str (equality only)
    - cancer_type: str (equality only)
    - stage: str (equality only)
    - biomarkers.EGFR: bool (equality only)
    - biomarkers.ALK: bool (equality only)
    - biomarkers.PD_L1: int/float (numeric)
    - lab_values.*: int/float (numeric)
    """
    model_config = ConfigDict(extra='forbid')
    
    field: str = Field(min_length=1)
    operator: Literal[">", "<", ">=", "<=", "==", "!="]
    value: Union[int, float, str, bool]
    
    FIELD_TYPES: ClassVar[dict] = {
        "age": (int, float),
        "gender": (str,),
        "cancer_type": (str,),
        "stage": (str,),
        "biomarkers.EGFR": (bool,),
        "biomarkers.ALK": (bool,),
        "biomarkers.PD_L1": (int, float),
        "lab_values.hb": (int, float),
        "lab_values.wbc": (int, float),
        "lab_values.creatinine": (int, float)
    }
    
    NUMERIC_OPERATORS: ClassVar[set] = {">", "<", ">=", "<="}
    EQUALITY_OPERATORS: ClassVar[set] = {"==", "!="}
    
    @field_validator('field')
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validate that field exists in Patient schema."""
        if not v or not v.strip():
            raise ValueError("field cannot be empty or whitespace")
        
        if v not in cls.FIELD_TYPES:
            raise ValueError(f"Unknown field: {v}. Must be one of {set(cls.FIELD_TYPES.keys())}")
        
        return v
    
    @model_validator(mode='after')
    def validate_field_value_type_compatibility(self) -> 'Rule':
        """Ensure value type matches field type and operator is valid for field type."""
        expected_types = self.FIELD_TYPES[self.field]
        
        if not isinstance(self.value, expected_types):
            raise ValueError(
                f"Field '{self.field}' expects type {expected_types}, "
                f"but got {type(self.value).__name__} with value {self.value}"
            )
        
        if self.operator in self.NUMERIC_OPERATORS:
            if not isinstance(self.value, (int, float)):
                raise ValueError(
                    f"Operator '{self.operator}' requires numeric value, "
                    f"but field '{self.field}' has value {self.value} of type {type(self.value).__name__}"
                )
        
        if self.field in {"gender", "cancer_type", "stage", "biomarkers.EGFR", "biomarkers.ALK"}:
            if self.operator in self.NUMERIC_OPERATORS:
                raise ValueError(
                    f"Field '{self.field}' is categorical and cannot use numeric operator '{self.operator}'. "
                    f"Use '==' or '!=' instead."
                )
        
        return self


class RequiredBiomarkers(BaseModel):
    """
    Biomarker requirements for trial eligibility.
    
    Semantics:
    - EGFR: None = not required, True = must be positive, False = must be negative
    - ALK: None = not required, True = must be positive, False = must be negative
    - PD_L1: None = not required, float = MINIMUM threshold (patient.PD_L1 >= trial.PD_L1)
    - EGFR_expression_min: None = not required, float = minimum expression level
    - ALK_expression_min: None = not required, float = minimum expression level
    
    Example: PD_L1=50.0 means patient must have PD_L1 >= 50.0%
    """
    model_config = ConfigDict(extra='forbid')
    
    EGFR: Optional[bool] = None
    ALK: Optional[bool] = None
    PD_L1: Optional[float] = Field(None, ge=0, le=100)
    EGFR_expression_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    ALK_expression_min: Optional[float] = Field(None, ge=0.0, le=1.0)


class DisallowedCondition(BaseModel):
    """Disallowed comorbidity with minimum severity threshold.
    
    Semantics:
    - min_severity="mild" means ANY severity is disallowed
    - min_severity="moderate" means only moderate/severe disallowed
    - min_severity="severe" means only severe disallowed
    """
    model_config = ConfigDict(extra='forbid')
    
    name: str
    min_severity: Literal["mild", "moderate", "severe"] = "mild"


class ClinicalTrial(BaseModel):
    """
    Strict clinical trial schema for programmatic eligibility.
    
    ALL fields required except where explicitly Optional.
    NO free-text criteria allowed.
    Minimum rule counts enforced.
    """
    model_config = ConfigDict(extra='forbid')
    
    trial_id: str = Field(min_length=1)
    cancer_type: Literal["lung cancer", "breast cancer", "colon cancer"]
    inclusion_criteria: list[Rule] = Field(min_length=3)
    exclusion_criteria: list[Rule] = Field(min_length=2)
    required_biomarkers: RequiredBiomarkers
    disallowed_conditions: list[DisallowedCondition] = Field(default_factory=list)
    required_prior_treatments: list[str] = Field(default_factory=list)
    forbidden_prior_treatments: list[str] = Field(default_factory=list)
    max_patients: int = Field(default=10, ge=1, le=100)
    enrolled_patients: int = Field(default=0, ge=0)
    days_until_deadline: int = Field(default=90, ge=0, le=365)
    trial_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @field_validator('trial_id')
    @classmethod
    def validate_trial_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("trial_id cannot be empty or whitespace")
        return v
    
    @field_validator('enrolled_patients')
    @classmethod
    def validate_enrolled(cls, v: int, info) -> int:
        max_p = info.data.get('max_patients', 10)
        if v > max_p:
            raise ValueError(f"enrolled_patients ({v}) cannot exceed max_patients ({max_p})")
        return v
    
    @property
    def has_capacity(self) -> bool:
        return self.enrolled_patients < self.max_patients
    
    @property
    def spots_remaining(self) -> int:
        return self.max_patients - self.enrolled_patients
    
    @property
    def is_urgent(self) -> bool:
        return self.days_until_deadline <= 14


def generate_random_trial(seed: int = None) -> ClinicalTrial:
    """
    Generate a random valid clinical trial with realistic constraints.
    
    Args:
        seed: Optional seed for reproducibility. If provided, output is deterministic.
    
    Returns:
        A valid ClinicalTrial instance with logically consistent rules.
    """
    if seed is not None:
        random.seed(seed)
    
    cancer_type = random.choice(["lung cancer", "breast cancer", "colon cancer"])
    
    inclusion_rules = []
    
    min_age = random.choice([18, 21])
    inclusion_rules.append(Rule(
        field="age",
        operator=">=",
        value=min_age
    ))
    
    max_age = random.choice([65, 70, 75, 80])
    inclusion_rules.append(Rule(
        field="age",
        operator="<=",
        value=max_age
    ))
    
    inclusion_rules.append(Rule(
        field="cancer_type",
        operator="==",
        value=cancer_type
    ))
    
    if random.random() > 0.5:
        min_hb = round(random.uniform(8.0, 10.0), 1)
        inclusion_rules.append(Rule(
            field="lab_values.hb",
            operator=">=",
            value=min_hb
        ))
    
    if random.random() > 0.5:
        max_creatinine = round(random.uniform(1.5, 2.5), 1)
        inclusion_rules.append(Rule(
            field="lab_values.creatinine",
            operator="<=",
            value=max_creatinine
        ))
    
    if random.random() > 0.5:
        min_wbc = random.choice([2500, 3000, 3500])
        inclusion_rules.append(Rule(
            field="lab_values.wbc",
            operator=">=",
            value=min_wbc
        ))
    
    exclusion_rules = []
    
    if random.random() > 0.3:
        age_excl_threshold = random.randint(max_age - 10, max_age - 1)
        exclusion_rules.append(Rule(
            field="age",
            operator=">",
            value=age_excl_threshold
        ))
    
    has_creatinine_inclusion = any(r.field == "lab_values.creatinine" for r in inclusion_rules)
    if has_creatinine_inclusion and random.random() > 0.3:
        incl_creatinine_max = next(r.value for r in inclusion_rules if r.field == "lab_values.creatinine")
        excl_creatinine = round(random.uniform(incl_creatinine_max * 0.7, incl_creatinine_max * 0.95), 1)
        exclusion_rules.append(Rule(
            field="lab_values.creatinine",
            operator=">",
            value=excl_creatinine
        ))
    
    has_hb_inclusion = any(r.field == "lab_values.hb" for r in inclusion_rules)
    if has_hb_inclusion and random.random() > 0.3:
        incl_hb_min = next(r.value for r in inclusion_rules if r.field == "lab_values.hb")
        excl_hb = round(random.uniform(incl_hb_min * 1.05, incl_hb_min * 1.3), 1)
        exclusion_rules.append(Rule(
            field="lab_values.hb",
            operator="<",
            value=excl_hb
        ))
    
    if random.random() > 0.5:
        exclusion_rules.append(Rule(
            field="stage",
            operator="==",
            value="IV"
        ))
    
    if len(exclusion_rules) < 2:
        has_wbc_inclusion = any(r.field == "lab_values.wbc" for r in inclusion_rules)
        if has_wbc_inclusion:
            incl_wbc_min = next(r.value for r in inclusion_rules if r.field == "lab_values.wbc")
            excl_wbc = int(incl_wbc_min * 1.1)
            exclusion_rules.append(Rule(
                field="lab_values.wbc",
                operator="<",
                value=excl_wbc
            ))
        else:
            exclusion_rules.append(Rule(
                field="lab_values.wbc",
                operator="<",
                value=2500
            ))
    
    if len(exclusion_rules) < 2:
        exclusion_rules.append(Rule(
            field="lab_values.hb",
            operator="<",
            value=7.5
        ))
    
    egfr_req = None
    alk_req = None
    pd_l1_req = None
    
    biomarker_choice = random.choice(["EGFR", "ALK", "PD_L1", "none", "multiple"])
    
    if biomarker_choice == "EGFR":
        egfr_req = True
    elif biomarker_choice == "ALK":
        alk_req = True
    elif biomarker_choice == "PD_L1":
        pd_l1_req = round(random.uniform(1.0, 50.0), 1)
    elif biomarker_choice == "multiple":
        if random.random() > 0.5:
            egfr_req = random.choice([True, False])
        if random.random() > 0.5:
            alk_req = random.choice([True, False])
        if random.random() > 0.5:
            pd_l1_req = round(random.uniform(1.0, 50.0), 1)
    
    # Sometimes add expression thresholds for biomarkers
    egfr_expr_min = None
    alk_expr_min = None
    if random.random() > 0.6 and egfr_req is True:
        egfr_expr_min = round(random.uniform(0.5, 0.8), 2)
    if random.random() > 0.6 and alk_req is True:
        alk_expr_min = round(random.uniform(0.5, 0.8), 2)

    all_conditions = ["hypertension", "diabetes", "COPD", "heart disease", "kidney disease", "liver disease"]
    num_disallowed = random.randint(0, 3)
    disallowed_names = random.sample(all_conditions, k=num_disallowed)
    disallowed = [
        DisallowedCondition(
            name=name,
            min_severity=random.choice(["mild", "moderate", "severe"])
        )
        for name in disallowed_names
    ]
    
    # Sometimes add prior treatment requirements
    VALID_TREATMENTS = [
        "chemotherapy", "immunotherapy", "radiation therapy",
        "surgery", "hormone therapy", "targeted therapy"
    ]
    if random.random() > 0.6:
        required_prior_treatments = random.sample(VALID_TREATMENTS, 1)
    else:
        required_prior_treatments = []
    if random.random() > 0.7:
        remaining = [t for t in VALID_TREATMENTS if t not in required_prior_treatments]
        forbidden_prior_treatments = random.sample(remaining, 1)
    else:
        forbidden_prior_treatments = []

    trial_id = f"TRIAL-{cancer_type.split()[0].upper()}-{random.randint(1000, 9999)}"
    
    max_patients = random.randint(3, 20)
    enrolled_patients = random.randint(0, max_patients)
    days_until_deadline = random.choice([3, 7, 14, 30, 60, 90, 120, 180, 365])
    trial_score = round(random.uniform(0.1, 1.0), 2)
    
    trial = ClinicalTrial(
        trial_id=trial_id,
        cancer_type=cancer_type,
        inclusion_criteria=inclusion_rules,
        exclusion_criteria=exclusion_rules,
        required_biomarkers=RequiredBiomarkers(
            EGFR=egfr_req,
            ALK=alk_req,
            PD_L1=pd_l1_req,
            EGFR_expression_min=egfr_expr_min,
            ALK_expression_min=alk_expr_min
        ),
        disallowed_conditions=disallowed,
        required_prior_treatments=required_prior_treatments,
        forbidden_prior_treatments=forbidden_prior_treatments,
        max_patients=max_patients,
        enrolled_patients=enrolled_patients,
        days_until_deadline=days_until_deadline,
        trial_score=trial_score
    )
    
    return trial
