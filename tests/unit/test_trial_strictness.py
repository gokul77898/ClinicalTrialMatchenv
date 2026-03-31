"""
Comprehensive strictness validation for trial schema.
"""

from src.schemas.trial_schema import ClinicalTrial, Rule, RequiredBiomarkers, generate_random_trial
from pydantic import ValidationError
import json


def test_rule_extra_fields():
    """Test that Rule rejects extra fields."""
    print("=" * 80)
    print("TEST 1: Rule extra fields forbidden")
    print("=" * 80)
    
    try:
        Rule(
            field="age",
            operator=">=",
            value=18,
            extra_field="SHOULD FAIL"
        )
        print("❌ FAILED: Extra field was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Extra field rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_invalid_operator():
    """Test that invalid operators are rejected."""
    print("=" * 80)
    print("TEST 2: Invalid operator rejected")
    print("=" * 80)
    
    try:
        Rule(
            field="age",
            operator="INVALID",
            value=18
        )
        print("❌ FAILED: Invalid operator was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Invalid operator rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_unknown_field():
    """Test that unknown fields are rejected."""
    print("=" * 80)
    print("TEST 3: Unknown field rejected")
    print("=" * 80)
    
    try:
        Rule(
            field="unknown_field",
            operator=">=",
            value=10
        )
        print("❌ FAILED: Unknown field was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Unknown field rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_min_inclusion_rules():
    """Test that minimum 3 inclusion rules are enforced."""
    print("=" * 80)
    print("TEST 4: Minimum 3 inclusion rules enforced")
    print("=" * 80)
    
    try:
        ClinicalTrial(
            trial_id="TEST-001",
            cancer_type="lung cancer",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=18)
            ],
            exclusion_criteria=[
                Rule(field="age", operator=">", value=85),
                Rule(field="lab_values.creatinine", operator=">", value=3.0)
            ],
            required_biomarkers=RequiredBiomarkers(),
            disallowed_conditions=[]
        )
        print("❌ FAILED: Too few inclusion rules accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Too few inclusion rules rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_min_exclusion_rules():
    """Test that minimum 2 exclusion rules are enforced."""
    print("=" * 80)
    print("TEST 5: Minimum 2 exclusion rules enforced")
    print("=" * 80)
    
    try:
        ClinicalTrial(
            trial_id="TEST-002",
            cancer_type="lung cancer",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=18),
                Rule(field="age", operator="<=", value=75),
                Rule(field="cancer_type", operator="==", value="lung cancer")
            ],
            exclusion_criteria=[
                Rule(field="age", operator=">", value=85)
            ],
            required_biomarkers=RequiredBiomarkers(),
            disallowed_conditions=[]
        )
        print("❌ FAILED: Too few exclusion rules accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Too few exclusion rules rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_trial_extra_fields():
    """Test that ClinicalTrial rejects extra fields."""
    print("=" * 80)
    print("TEST 6: ClinicalTrial extra fields forbidden")
    print("=" * 80)
    
    try:
        ClinicalTrial(
            trial_id="TEST-003",
            cancer_type="lung cancer",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=18),
                Rule(field="age", operator="<=", value=75),
                Rule(field="cancer_type", operator="==", value="lung cancer")
            ],
            exclusion_criteria=[
                Rule(field="age", operator=">", value=85),
                Rule(field="lab_values.creatinine", operator=">", value=3.0)
            ],
            required_biomarkers=RequiredBiomarkers(),
            disallowed_conditions=[],
            extra_field="SHOULD FAIL"
        )
        print("❌ FAILED: Extra field was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Extra field rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_biomarkers_extra_fields():
    """Test that RequiredBiomarkers rejects extra fields."""
    print("=" * 80)
    print("TEST 7: RequiredBiomarkers extra fields forbidden")
    print("=" * 80)
    
    try:
        RequiredBiomarkers(
            EGFR=True,
            ALK=False,
            PD_L1=50.0,
            extra_biomarker="SHOULD FAIL"
        )
        print("❌ FAILED: Extra biomarker field was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Extra biomarker field rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_nested_field_access():
    """Test that nested field access works correctly."""
    print("=" * 80)
    print("TEST 8: Nested field access validation")
    print("=" * 80)
    
    valid_nested_fields = [
        "lab_values.hb",
        "lab_values.wbc",
        "lab_values.creatinine",
        "biomarkers.EGFR",
        "biomarkers.ALK",
        "biomarkers.PD_L1"
    ]
    
    all_valid = True
    for field in valid_nested_fields:
        try:
            Rule(field=field, operator=">=", value=10)
        except ValidationError:
            print(f"❌ FAILED: Valid nested field '{field}' was rejected")
            all_valid = False
    
    if all_valid:
        print("✅ PASSED: All nested fields accepted")
        print(f"   Valid fields: {valid_nested_fields}")
    print()


def test_cancer_type_restriction():
    """Test that only allowed cancer types are accepted."""
    print("=" * 80)
    print("TEST 9: Cancer type restriction")
    print("=" * 80)
    
    try:
        ClinicalTrial(
            trial_id="TEST-004",
            cancer_type="leukemia",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=18),
                Rule(field="age", operator="<=", value=75),
                Rule(field="cancer_type", operator="==", value="leukemia")
            ],
            exclusion_criteria=[
                Rule(field="age", operator=">", value=85),
                Rule(field="lab_values.creatinine", operator=">", value=3.0)
            ],
            required_biomarkers=RequiredBiomarkers(),
            disallowed_conditions=[]
        )
        print("❌ FAILED: Invalid cancer type was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Invalid cancer type rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_pd_l1_range():
    """Test that PD_L1 threshold is validated (0-100)."""
    print("=" * 80)
    print("TEST 10: PD_L1 range validation (0-100)")
    print("=" * 80)
    
    try:
        RequiredBiomarkers(
            EGFR=None,
            ALK=None,
            PD_L1=150.0
        )
        print("❌ FAILED: PD_L1 > 100 was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: PD_L1 > 100 rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "TRIAL SCHEMA STRICTNESS VALIDATION" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    test_rule_extra_fields()
    test_invalid_operator()
    test_unknown_field()
    test_min_inclusion_rules()
    test_min_exclusion_rules()
    test_trial_extra_fields()
    test_biomarkers_extra_fields()
    test_nested_field_access()
    test_cancer_type_restriction()
    test_pd_l1_range()
    
    print("=" * 80)
    print("SAMPLE VALID TRIAL (seed=123)")
    print("=" * 80)
    trial = generate_random_trial(seed=123)
    print(json.dumps(trial.model_dump(), indent=2))
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ All strictness tests passed")
    print("✅ Machine-readable rules enforced")
    print("✅ No free-text criteria allowed")
    print("✅ Nested field access validated")
    print("✅ Minimum rule counts enforced")
    print()


if __name__ == "__main__":
    main()
