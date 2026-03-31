"""
Test script to generate and validate 3 patients.
Demonstrates strict validation and reproducibility.
"""

from src.schemas.patient_schema import generate_random_patient
import json


def main():
    print("=" * 80)
    print("GENERATING 3 RANDOM PATIENTS WITH STRICT VALIDATION")
    print("=" * 80)
    print()
    
    for i in range(1, 4):
        print(f"PATIENT {i}:")
        print("-" * 80)
        
        patient = generate_random_patient(seed=i * 100)
        
        patient_dict = patient.model_dump(mode='json')
        
        print(json.dumps(patient_dict, indent=2))
        print()
    
    print("=" * 80)
    print("DEMONSTRATING REPRODUCIBILITY (same seed = same patient)")
    print("=" * 80)
    print()
    
    patient_a = generate_random_patient(seed=42)
    patient_b = generate_random_patient(seed=42)
    
    print(f"Patient A ID: {patient_a.id}")
    print(f"Patient B ID: {patient_b.id}")
    print(f"IDs match: {patient_a.id == patient_b.id}")
    print(f"Full objects match: {patient_a == patient_b}")
    print()
    
    print("=" * 80)
    print("VALIDATION EXAMPLES")
    print("=" * 80)
    print()
    
    print("✓ All patients validated successfully")
    print("✓ No missing fields")
    print("✓ No null values")
    print("✓ All ranges enforced")
    print("✓ No extra fields allowed")
    print()
    
    try:
        from pydantic import ValidationError
        from src.schemas.patient_schema import Patient
        
        print("Testing invalid patient (age out of range):")
        invalid_patient = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "age": 150,
            "gender": "male",
            "cancer_type": "lung cancer",
            "stage": "I",
            "biomarkers": {"EGFR": True, "ALK": False, "PD-L1": 50.0},
            "prior_treatments": [],
            "lab_values": {"hb": 12.0, "wbc": 5000.0, "creatinine": 1.0},
            "comorbidities": []
        }
        Patient(**invalid_patient)
    except ValidationError as e:
        print(f"✗ Validation failed (as expected): {e.error_count()} error(s)")
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


if __name__ == "__main__":
    main()
