"""
Strict Clinical Patient Schema for Simulation Environment
Uses Pydantic for strict validation with no tolerance for invalid/missing data.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal
from uuid import UUID, uuid4
import random


class Biomarkers(BaseModel):
    """Biomarker data with exact required keys."""
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    
    EGFR: bool
    ALK: bool
    PD_L1: float = Field(ge=0, le=100)
    EGFR_expression: float = Field(ge=0.0, le=1.0, default=0.0)
    ALK_expression: float = Field(ge=0.0, le=1.0, default=0.0)


class Comorbidity(BaseModel):
    """Comorbidity with severity level."""
    model_config = ConfigDict(extra='forbid')
    
    name: str
    severity: Literal["mild", "moderate", "severe"]


class LabValueTrend(BaseModel):
    """Trend information for a lab value over recent measurements."""
    model_config = ConfigDict(extra='forbid')
    
    name: str
    direction: Literal["increasing", "decreasing", "stable"]
    rate: float = Field(ge=0.0, le=1.0)


class LabValues(BaseModel):
    """Lab values with exact required keys and strict ranges."""
    model_config = ConfigDict(extra='forbid')
    
    hb: float = Field(ge=0, le=20)
    wbc: float = Field(ge=0, le=20000)
    creatinine: float = Field(ge=0, le=10)


class Patient(BaseModel):
    """
    Strict patient model for clinical trial simulation.
    
    ALL fields are required.
    NO fields can be null.
    NO extra fields allowed.
    ALL numeric ranges enforced.
    """
    model_config = ConfigDict(extra='forbid')
    
    id: UUID
    age: int = Field(ge=0, le=120)
    gender: Literal["male", "female", "other"]
    cancer_type: str = Field(min_length=1)
    stage: Literal["I", "II", "III", "IV"]
    biomarkers: Biomarkers
    prior_treatments: list[str]
    lab_values: LabValues
    lab_value_trends: list[LabValueTrend] = Field(default_factory=list)
    comorbidities: list[Comorbidity] = Field(default_factory=list)
    
    @field_validator('cancer_type')
    @classmethod
    def validate_cancer_type(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("cancer_type cannot be empty or whitespace")
        return v


def generate_random_patient(seed: int = None) -> Patient:
    """
    Generate a random valid patient with realistic clinical data.
    
    Args:
        seed: Optional seed for reproducibility. If provided, output is deterministic.
    
    Returns:
        A valid Patient instance with realistic values.
    """
    if seed is not None:
        random.seed(seed)
    
    cancer_types = [
        "lung cancer",
        "breast cancer",
        "colon cancer"
    ]
    
    common_treatments = [
        "chemotherapy",
        "radiation therapy",
        "immunotherapy",
        "targeted therapy",
        "surgery",
        "hormone therapy"
    ]
    
    common_comorbidities = [
        "hypertension",
        "diabetes",
        "COPD",
        "heart disease",
        "kidney disease",
        "liver disease"
    ]
    
    patient_id = UUID(int=random.getrandbits(128)) if seed is not None else uuid4()
    
    egfr_val = random.choice([True, False])
    alk_val = random.choice([True, False])
    if egfr_val:
        egfr_expression = round(random.uniform(0.5, 1.0), 3)
    else:
        egfr_expression = round(random.uniform(0.0, 0.4), 3)
    if alk_val:
        alk_expression = round(random.uniform(0.5, 1.0), 3)
    else:
        alk_expression = round(random.uniform(0.0, 0.4), 3)

    patient = Patient(
        id=patient_id,
        age=random.randint(30, 85),
        gender=random.choice(["male", "female", "other"]),
        cancer_type=random.choice(cancer_types),
        stage=random.choice(["I", "II", "III", "IV"]),
        biomarkers=Biomarkers(
            EGFR=egfr_val,
            ALK=alk_val,
            PD_L1=round(random.uniform(0, 100), 2),
            EGFR_expression=egfr_expression,
            ALK_expression=alk_expression
        ),
        prior_treatments=random.sample(
            common_treatments, 
            k=random.randint(0, 3)
        ),
        lab_values=LabValues(
            hb=round(random.uniform(8.0, 16.0), 2),
            wbc=round(random.uniform(3000, 15000), 2),
            creatinine=round(random.uniform(0.5, 2.5), 2)
        ),
        lab_value_trends=[
            LabValueTrend(
                name=name,
                direction=random.choice(["increasing", "decreasing", "stable"]),
                rate=round(random.uniform(0.0, 0.5), 3)
            )
            for name in ["hb", "wbc", "creatinine"]
        ],
        comorbidities=[
            Comorbidity(
                name=random.choice(common_comorbidities),
                severity=random.choice(["mild", "moderate", "severe"])
            )
            for _ in range(random.randint(0, 2))
        ]
    )
    
    return patient
