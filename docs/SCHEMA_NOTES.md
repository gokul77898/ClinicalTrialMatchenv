# Clinical Trial Schema - Implementation Notes

## ✅ Phase 1 Complete - Schema Ready for Eligibility Engine

### Patient Schema
- **File:** `patient_schema.py`
- **Status:** Production-ready
- Strict Pydantic validation with `extra='forbid'`
- All fields required, no nulls allowed
- Reproducible generation with seeds
- Cancer types: lung, breast, colon (medically consistent staging)

### Trial Schema
- **File:** `trial_schema.py`
- **Status:** Production-ready
- Machine-readable rules (NO free-text criteria)
- Field-type validation enforced
- Nested field access validated structurally
- Logical consistency: exclusions cut INTO inclusion space

---

## ⚠️ CRITICAL EDGE CASES (For Eligibility Engine Testing)

### 1. Overlapping Inclusion/Exclusion Rules

**Example - Valid Narrow Band:**
```
Inclusion: age >= 21 AND age <= 80
Exclusion: age > 75
→ Valid band: 21-75 (ages 76-80 excluded)
```
✅ This is correct behavior.

**Example - DANGEROUS Annihilation:**
```
Inclusion: age >= 21 AND age <= 80
Exclusion: age > 20
→ Valid band: NONE (everything excluded)
```
❌ This would make 0 patients eligible.

**Testing Strategy:**
- When running eligibility matching, track:
  - Number of eligible patients per trial
  - If many trials have 0 eligible patients → generator issue
- Exclusion should **REDUCE** space, not **ANNIHILATE** it

**Current Generator Status:**
- Age exclusions: `random.randint(max_age - 10, max_age - 1)` ✅
- Lab value exclusions: Based on inclusion thresholds ✅
- Risk is LOW but not zero (random combinations could still create edge cases)

---

### 2. FIELD_TYPES Mapping Maintenance

**Current Implementation:**
```python
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
```

**Risk:**
- Manually defined → can drift from Patient schema
- If Patient schema adds/removes fields → FIELD_TYPES won't update automatically
- Silent mismatch → broken eligibility engine

**Future Improvement (Post-Phase 1):**
```python
# Auto-derive from Patient model using introspection
def build_field_types_from_patient_schema():
    """Introspect Patient model to build FIELD_TYPES mapping."""
    from patient_schema import Patient, Biomarkers, LabValues
    
    field_types = {}
    # Extract from Patient.__annotations__
    # Extract nested from Biomarkers.__annotations__
    # Extract nested from LabValues.__annotations__
    return field_types
```

**Status:** Not blocking Phase 1, but flagged for production hardening.

---

## 🎯 Next Phase: Eligibility Engine

### Requirements
1. Implement rule evaluation logic
2. Handle nested field access (`lab_values.hb`, `biomarkers.PD_L1`)
3. Apply inclusion criteria (AND logic)
4. Apply exclusion criteria (disqualify if ANY match)
5. Check biomarker requirements
6. Check disallowed conditions
7. Return match score or boolean

### Edge Cases to Test
- [ ] Trial with narrow valid band (inclusion + exclusion overlap)
- [ ] Trial that annihilates all valid space (0 eligible patients)
- [ ] Patient at exact threshold boundaries
- [ ] Patient with missing biomarker data (None values)
- [ ] Multiple trials, verify at least some patients match

### PD_L1 Semantics (CRITICAL)
```
Trial: PD_L1 = 50.0
Patient: PD_L1 = 60.0
→ ELIGIBLE (patient.PD_L1 >= trial.PD_L1)

Trial: PD_L1 = 50.0
Patient: PD_L1 = 40.0
→ NOT ELIGIBLE (patient.PD_L1 < trial.PD_L1)
```

---

## 📊 Validation Summary

### Patient Schema Tests
- ✅ 8/8 strictness tests passed
- ✅ PD_L1 field name consistency (no drift)
- ✅ UUID validity (reproducible)
- ✅ Cancer types medically consistent
- ✅ Extra fields forbidden at all levels

### Trial Schema Tests
- ✅ 10/10 strictness tests passed
- ✅ Type safety enforced (age >= "high" rejected)
- ✅ Categorical fields reject numeric operators
- ✅ Nested field validation structural
- ✅ Minimum rule counts enforced

### Audit Fix Tests
- ✅ 10/10 audit tests passed
- ✅ Logical consistency (exclusions cut into inclusion)
- ✅ Field-type mapping complete
- ✅ PD_L1 threshold semantics documented

---

## 📁 File Structure

```
ClinicalTrialMatchEnv/
├── patient_schema.py          # Patient model + generator
├── trial_schema.py            # Trial model + Rule + generator
├── test_patient_generation.py # Patient tests
├── test_strictness.py         # Patient strictness tests
├── test_trial_generation.py   # Trial tests
├── test_trial_strictness.py   # Trial strictness tests
├── test_audit_fixes.py        # Audit validation tests
├── requirements.txt           # pydantic>=2.0.0
├── README.md                  # Usage documentation
└── SCHEMA_NOTES.md           # This file
```

---

## 🚀 Ready for Eligibility Engine Implementation

All critical issues addressed. Schema is production-ready for Phase 1.
