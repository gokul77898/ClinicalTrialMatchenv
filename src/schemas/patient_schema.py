"""
Clinical Patient Schema for Simulation Environment

Supports TWO MODES:
- STRICT_MODE: 100% Phase 1 spec compliance (no nulls, strict enums)
- REALISTIC_MODE: Advanced simulation (missing data, conflicts, trends)

Mode is controlled by src/config.py
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import Literal, Optional, Union
from uuid import UUID, uuid4
import random

from src.config import (
    STRICT_MODE,
    REALISTIC_MODE,
    ALLOW_MISSING_LAB_VALUES,
    ALLOW_CONFLICTING_FIELDS,
    ALLOW_TEMPORAL_TRENDS,
    CANCER_TYPE_STRICT_ENUM
)


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


if STRICT_MODE:
    class LabValues(BaseModel):
        """Lab values - STRICT_MODE: All values required, no nulls."""
        model_config = ConfigDict(extra='forbid')
        
        hb: float = Field(ge=0, le=20)
        wbc: float = Field(ge=0, le=20000)
        creatinine: float = Field(ge=0, le=10)

elif REALISTIC_MODE:
    class LabValues(BaseModel):
        """Lab values - REALISTIC_MODE: Values may be None (unknown)."""
        model_config = ConfigDict(extra='forbid')
        
        hb: Optional[float] = Field(None, ge=0, le=20)
        wbc: Optional[float] = Field(None, ge=0, le=20000)
        creatinine: Optional[float] = Field(None, ge=0, le=10)

        @property
        def hb_unknown(self) -> bool:
            return self.hb is None

        @property
        def wbc_unknown(self) -> bool:
            return self.wbc is None

        @property
        def creatinine_unknown(self) -> bool:
            return self.creatinine is None


if STRICT_MODE:
    class Patient(BaseModel):
        """
        STRICT_MODE Patient: 100% Phase 1 spec compliance.
        
        - ALL fields required
        - NO null values
        - cancer_type is strict enum
        - NO advanced features (trends, conflicts, freshness)
        """
        model_config = ConfigDict(extra='forbid')
        
        id: UUID
        age: int = Field(ge=0, le=120)
        gender: Literal["male", "female", "other"]
        cancer_type: Literal["lung cancer", "breast cancer", "colon cancer"]
        stage: Literal["I", "II", "III", "IV"]
        biomarkers: Biomarkers
        prior_treatments: list[str]
        lab_values: LabValues
        comorbidities: list[Comorbidity] = Field(default_factory=list)

elif REALISTIC_MODE:
    class Patient(BaseModel):
        """
        REALISTIC_MODE Patient: Advanced simulation features.
        
        - Lab values may be None (unknown)
        - Includes temporal trends
        - Includes conflicting fields
        - Includes data freshness tracking
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
        conflicting_fields: dict = Field(default_factory=dict)
        data_freshness_days: int = Field(default=0, ge=0, le=365)
        
        @field_validator('cancer_type')
        @classmethod
        def validate_cancer_type(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("cancer_type cannot be empty or whitespace")
            # In REALISTIC_MODE, allow any cancer type string
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

    # Lab values - mode-dependent
    if STRICT_MODE:
        # STRICT_MODE: All lab values MUST be present (no None)
        hb_val = round(random.uniform(8.0, 16.0), 2)
        wbc_val = round(random.uniform(3000, 15000), 2)
        creatinine_val = round(random.uniform(0.5, 2.5), 2)
    elif REALISTIC_MODE:
        # REALISTIC_MODE: 15% chance of missing lab values
        from src.config import MISSING_LAB_VALUE_PROBABILITY
        hb_val = None if random.random() < MISSING_LAB_VALUE_PROBABILITY else round(
            random.uniform(8.0, 16.0), 2
        )
        wbc_val = None if random.random() < MISSING_LAB_VALUE_PROBABILITY else round(
            random.uniform(3000, 15000), 2
        )
        creatinine_val = None if random.random() < MISSING_LAB_VALUE_PROBABILITY else round(
            random.uniform(0.5, 2.5), 2
        )

    patient_stage = random.choice(["I", "II", "III", "IV"])

    # Conflicting fields - only in REALISTIC_MODE
    if REALISTIC_MODE and ALLOW_CONFLICTING_FIELDS:
        from src.config import CONFLICTING_FIELD_PROBABILITY
        if random.random() < CONFLICTING_FIELD_PROBABILITY:
            conflicting_fields = {
                "stage": {
                    "reported": patient_stage,
                    "notes_say": "II" if patient_stage == "III" else "III",
                    "confidence": "low"
                }
            }
        else:
            conflicting_fields = {}
        data_freshness_days = random.randint(0, 180)
    else:
        conflicting_fields = {}
        data_freshness_days = 0

    # Build patient - mode-dependent fields
    comorbidities = [
        Comorbidity(
            name=random.choice(common_comorbidities),
            severity=random.choice(["mild", "moderate", "severe"])
        )
        for _ in range(random.randint(0, 2))
    ]
    
    if STRICT_MODE:
        # STRICT_MODE: No advanced features
        patient = Patient(
            id=patient_id,
            age=random.randint(30, 85),
            gender=random.choice(["male", "female", "other"]),
            cancer_type=random.choice(cancer_types),
            stage=patient_stage,
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
                hb=hb_val,
                wbc=wbc_val,
                creatinine=creatinine_val
            ),
            comorbidities=comorbidities
        )
    
    elif REALISTIC_MODE:
        # REALISTIC_MODE: Include all advanced features
        patient = Patient(
            id=patient_id,
            age=random.randint(30, 85),
            gender=random.choice(["male", "female", "other"]),
            cancer_type=random.choice(cancer_types),
            stage=patient_stage,
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
                hb=hb_val,
                wbc=wbc_val,
                creatinine=creatinine_val
            ),
            lab_value_trends=[
                LabValueTrend(
                    name=name,
                    direction=random.choice(["increasing", "decreasing", "stable"]),
                    rate=round(random.uniform(0.0, 0.5), 3)
                )
                for name in ["hb", "wbc", "creatinine"]
            ],
            comorbidities=comorbidities,
            conflicting_fields=conflicting_fields,
            data_freshness_days=data_freshness_days
        )
    
    return patient
