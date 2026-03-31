"""
Test script to validate ALL strictness requirements.
"""

from src.schemas.patient_schema import generate_random_patient, Patient, Biomarkers, LabValues
from pydantic import ValidationError
import json


def test_extra_fields_forbidden():
    """Test that extra fields are rejected."""
    print("=" * 80)
    print("TEST 1: Extra fields forbidden")
    print("=" * 80)
    
    try:
        Patient(
            id="550e8400-e29b-41d4-a716-446655440000",
            age=50,
            gender="male",
            cancer_type="lung cancer",
            stage="I",
            biomarkers={"EGFR": True, "ALK": False, "PD_L1": 50.0},
            prior_treatments=[],
            lab_values={"hb": 12.0, "wbc": 5000.0, "creatinine": 1.0},
            comorbidities=[],
            extra_field="SHOULD FAIL"
        )
        print("❌ FAILED: Extra field was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Extra field rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_missing_fields():
    """Test that missing fields are rejected."""
    print("=" * 80)
    print("TEST 2: Missing fields forbidden")
    print("=" * 80)
    
    try:
        Patient(
            id="550e8400-e29b-41d4-a716-446655440000",
            age=50,
            gender="male",
            cancer_type="lung cancer",
            stage="I",
            biomarkers={"EGFR": True, "ALK": False, "PD_L1": 50.0},
            prior_treatments=[],
            lab_values={"hb": 12.0, "wbc": 5000.0, "creatinine": 1.0}
        )
        print("❌ FAILED: Missing field was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Missing field rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_null_values():
    """Test that null values are rejected."""
    print("=" * 80)
    print("TEST 3: Null values forbidden")
    print("=" * 80)
    
    try:
        Patient(
            id="550e8400-e29b-41d4-a716-446655440000",
            age=50,
            gender="male",
            cancer_type="lung cancer",
            stage="I",
            biomarkers=None,
            prior_treatments=[],
            lab_values={"hb": 12.0, "wbc": 5000.0, "creatinine": 1.0},
            comorbidities=[]
        )
        print("❌ FAILED: Null value was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Null value rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def test_pd_l1_consistency():
    """Test that PD_L1 is consistent (no PD-L1 drift)."""
    print("=" * 80)
    print("TEST 4: PD_L1 field name consistency")
    print("=" * 80)
    
    patient = generate_random_patient(seed=999)
    patient_dict = patient.model_dump()
    
    if "PD_L1" in patient_dict["biomarkers"]:
        print("✅ PASSED: Field is 'PD_L1' (consistent)")
    elif "PD-L1" in patient_dict["biomarkers"]:
        print("❌ FAILED: Field is 'PD-L1' (inconsistent)")
    else:
        print("❌ FAILED: PD_L1 field not found")
    
    print(f"   Biomarkers keys: {list(patient_dict['biomarkers'].keys())}")
    print()


def test_uuid_validity():
    """Test that UUIDs are valid UUID4 format."""
    print("=" * 80)
    print("TEST 5: UUID validity")
    print("=" * 80)
    
    from uuid import UUID
    
    for i in range(5):
        patient = generate_random_patient(seed=i)
        try:
            uuid_obj = UUID(str(patient.id))
            print(f"✅ Patient {i}: Valid UUID - {patient.id}")
        except ValueError:
            print(f"❌ Patient {i}: Invalid UUID - {patient.id}")
    print()


def test_cancer_types():
    """Test that only valid cancer types are generated."""
    print("=" * 80)
    print("TEST 6: Cancer types (lung/breast/colon only)")
    print("=" * 80)
    
    valid_cancers = {"lung cancer", "breast cancer", "colon cancer"}
    invalid_found = []
    
    for i in range(20):
        patient = generate_random_patient(seed=i)
        if patient.cancer_type not in valid_cancers:
            invalid_found.append(patient.cancer_type)
    
    if invalid_found:
        print(f"❌ FAILED: Found invalid cancer types: {set(invalid_found)}")
    else:
        print("✅ PASSED: Only lung/breast/colon cancer types generated")
    print()


def test_biomarker_extra_fields():
    """Test that biomarkers reject extra fields."""
    print("=" * 80)
    print("TEST 7: Biomarkers extra fields forbidden")
    print("=" * 80)
    
    try:
        Biomarkers(
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


def test_lab_values_extra_fields():
    """Test that lab_values reject extra fields."""
    print("=" * 80)
    print("TEST 8: Lab values extra fields forbidden")
    print("=" * 80)
    
    try:
        LabValues(
            hb=12.0,
            wbc=5000.0,
            creatinine=1.0,
            extra_lab="SHOULD FAIL"
        )
        print("❌ FAILED: Extra lab value field was accepted (BAD)")
    except ValidationError as e:
        print("✅ PASSED: Extra lab value field rejected")
        print(f"   Error: {e.errors()[0]['msg']}")
    print()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "STRICTNESS VALIDATION SUITE" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    test_extra_fields_forbidden()
    test_missing_fields()
    test_null_values()
    test_pd_l1_consistency()
    test_uuid_validity()
    test_cancer_types()
    test_biomarker_extra_fields()
    test_lab_values_extra_fields()
    
    print("=" * 80)
    print("SAMPLE OUTPUT (seed=42)")
    print("=" * 80)
    patient = generate_random_patient(seed=42)
    print(json.dumps(patient.model_dump(), indent=2, default=str))
    print()


if __name__ == "__main__":
    main()
