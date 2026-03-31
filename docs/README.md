# Clinical Patient Schema - Simulation Environment

Strict patient data model with Pydantic validation for clinical trial simulations.

## Features

✅ **Zero tolerance for invalid data**
- All fields required (no nulls, no missing values)
- Strict type enforcement
- Numeric range validation
- No extra fields allowed

✅ **Reproducible patient generation**
- Deterministic output when seed is provided
- Realistic clinical value ranges

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Random Patients

```python
from patient_schema import generate_random_patient

# Random patient
patient = generate_random_patient()

# Reproducible patient (same seed = same output)
patient = generate_random_patient(seed=42)
```

### Run Test Script

```bash
python test_patient_generation.py
```

## Schema

### Patient Model

| Field | Type | Constraints |
|-------|------|-------------|
| `id` | UUID | Valid UUID format |
| `age` | int | 0–120 |
| `gender` | enum | "male", "female", "other" |
| `cancer_type` | string | Non-empty |
| `stage` | enum | "I", "II", "III", "IV" |
| `biomarkers` | object | See below |
| `prior_treatments` | list[str] | Can be empty, not null |
| `lab_values` | object | See below |
| `comorbidities` | list[str] | Can be empty, not null |

### Biomarkers

| Field | Type | Constraints |
|-------|------|-------------|
| `EGFR` | bool | - |
| `ALK` | bool | - |
| `PD-L1` | float | 0–100 |

### Lab Values

| Field | Type | Constraints |
|-------|------|-------------|
| `hb` | float | 0–20 |
| `wbc` | float | 0–20000 |
| `creatinine` | float | 0–10 |

## Validation Examples

```python
from patient_schema import Patient
from pydantic import ValidationError

# ✗ This will fail (age out of range)
try:
    Patient(
        id="550e8400-e29b-41d4-a716-446655440000",
        age=150,  # Invalid!
        ...
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# ✗ This will fail (missing field)
try:
    Patient(
        id="550e8400-e29b-41d4-a716-446655440000",
        age=50,
        # Missing gender field!
        ...
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# ✗ This will fail (extra field)
try:
    Patient(
        id="550e8400-e29b-41d4-a716-446655440000",
        age=50,
        gender="male",
        extra_field="not allowed",  # Invalid!
        ...
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Generator Realistic Ranges

- **Age**: 30–85
- **Hemoglobin (hb)**: 8–16 g/dL
- **WBC**: 3000–15000 cells/μL
- **Creatinine**: 0.5–2.5 mg/dL
- **PD-L1**: 0–100%
- **Cancer types**: lung, breast, colon, prostate, melanoma, pancreatic, leukemia, lymphoma
- **Stages**: I, II, III, IV (random)
