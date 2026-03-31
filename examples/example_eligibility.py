"""
Example usage of the eligibility engine with 3 test cases.

CASE 1: ELIGIBLE - Patient satisfies all criteria
CASE 2: INCLUSION FAIL - Patient fails inclusion rules
CASE 3: EXCLUSION TRIGGERED - Patient passes inclusion but triggers exclusion
"""

from src.schemas.patient_schema import Patient, Biomarkers, LabValues
from src.schemas.trial_schema import ClinicalTrial, Rule, RequiredBiomarkers
from src.engine.eligibility_engine import is_eligible, check_inclusion, check_exclusion
from uuid import uuid4


def print_case(case_num: int, title: str):
    """Print formatted case header."""
    print("\n" + "=" * 80)
    print(f"CASE {case_num}: {title}")
    print("=" * 80)


def main():
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "ELIGIBILITY ENGINE EXAMPLES" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    print_case(1, "ELIGIBLE - Patient satisfies all criteria")
    
    patient_eligible = Patient(
        id=uuid4(),
        age=55,
        gender="male",
        cancer_type="lung cancer",
        stage="II",
        biomarkers=Biomarkers(
            EGFR=True,
            ALK=False,
            PD_L1=60.0
        ),
        prior_treatments=["chemotherapy"],
        lab_values=LabValues(
            hb=12.5,
            wbc=7000.0,
            creatinine=1.2
        ),
        comorbidities=["hypertension"]
    )
    
    trial_1 = ClinicalTrial(
        trial_id="TRIAL-LUNG-001",
        cancer_type="lung cancer",
        inclusion_criteria=[
            Rule(field="age", operator=">=", value=18),
            Rule(field="age", operator="<=", value=75),
            Rule(field="cancer_type", operator="==", value="lung cancer"),
            Rule(field="lab_values.hb", operator=">=", value=10.0)
        ],
        exclusion_criteria=[
            Rule(field="age", operator=">", value=80),
            Rule(field="lab_values.creatinine", operator=">", value=2.0)
        ],
        required_biomarkers=RequiredBiomarkers(
            EGFR=True,
            ALK=None,
            PD_L1=50.0
        ),
        disallowed_conditions=["kidney disease", "liver disease"]
    )
    
    eligible = is_eligible(patient_eligible, trial_1)
    print(f"\nPatient: {patient_eligible.age}yo {patient_eligible.gender}, {patient_eligible.cancer_type}, stage {patient_eligible.stage}")
    print(f"  Lab values: hb={patient_eligible.lab_values.hb}, wbc={patient_eligible.lab_values.wbc}, creatinine={patient_eligible.lab_values.creatinine}")
    print(f"  Biomarkers: EGFR={patient_eligible.biomarkers.EGFR}, ALK={patient_eligible.biomarkers.ALK}, PD_L1={patient_eligible.biomarkers.PD_L1}")
    print(f"  Comorbidities: {patient_eligible.comorbidities}")
    print(f"\nTrial: {trial_1.trial_id}")
    print(f"  Inclusion: age 18-75, lung cancer, hb >= 10.0")
    print(f"  Exclusion: age > 80, creatinine > 2.0")
    print(f"  Biomarkers: EGFR=True, PD_L1 >= 50.0")
    print(f"  Disallowed: kidney disease, liver disease")
    print(f"\n✅ Eligible: {eligible}")
    
    print_case(2, "INCLUSION FAIL - Patient fails inclusion rules")
    
    patient_too_young = Patient(
        id=uuid4(),
        age=16,
        gender="female",
        cancer_type="lung cancer",
        stage="I",
        biomarkers=Biomarkers(
            EGFR=True,
            ALK=False,
            PD_L1=55.0
        ),
        prior_treatments=[],
        lab_values=LabValues(
            hb=13.0,
            wbc=6500.0,
            creatinine=0.9
        ),
        comorbidities=[]
    )
    
    trial_2 = ClinicalTrial(
        trial_id="TRIAL-LUNG-002",
        cancer_type="lung cancer",
        inclusion_criteria=[
            Rule(field="age", operator=">=", value=18),
            Rule(field="age", operator="<=", value=70),
            Rule(field="cancer_type", operator="==", value="lung cancer")
        ],
        exclusion_criteria=[
            Rule(field="stage", operator="==", value="IV"),
            Rule(field="lab_values.hb", operator="<", value=8.0)
        ],
        required_biomarkers=RequiredBiomarkers(
            EGFR=None,
            ALK=None,
            PD_L1=None
        ),
        disallowed_conditions=[]
    )
    
    eligible = is_eligible(patient_too_young, trial_2)
    inclusion_pass = check_inclusion(patient_too_young, trial_2)
    
    print(f"\nPatient: {patient_too_young.age}yo {patient_too_young.gender}, {patient_too_young.cancer_type}, stage {patient_too_young.stage}")
    print(f"  Lab values: hb={patient_too_young.lab_values.hb}, wbc={patient_too_young.lab_values.wbc}, creatinine={patient_too_young.lab_values.creatinine}")
    print(f"\nTrial: {trial_2.trial_id}")
    print(f"  Inclusion: age 18-70, lung cancer")
    print(f"  Exclusion: stage IV, hb < 8.0")
    print(f"\n❌ Eligible: {eligible} (Inclusion Failed)")
    print(f"   Reason: Patient age {patient_too_young.age} < minimum age 18")
    
    print_case(3, "EXCLUSION TRIGGERED - Patient passes inclusion but triggers exclusion")
    
    patient_high_creatinine = Patient(
        id=uuid4(),
        age=62,
        gender="male",
        cancer_type="colon cancer",
        stage="III",
        biomarkers=Biomarkers(
            EGFR=False,
            ALK=False,
            PD_L1=25.0
        ),
        prior_treatments=["surgery", "chemotherapy"],
        lab_values=LabValues(
            hb=11.0,
            wbc=5500.0,
            creatinine=2.8
        ),
        comorbidities=["diabetes"]
    )
    
    trial_3 = ClinicalTrial(
        trial_id="TRIAL-COLON-003",
        cancer_type="colon cancer",
        inclusion_criteria=[
            Rule(field="age", operator=">=", value=21),
            Rule(field="age", operator="<=", value=75),
            Rule(field="cancer_type", operator="==", value="colon cancer"),
            Rule(field="lab_values.hb", operator=">=", value=9.0)
        ],
        exclusion_criteria=[
            Rule(field="lab_values.creatinine", operator=">", value=2.5),
            Rule(field="stage", operator="==", value="IV")
        ],
        required_biomarkers=RequiredBiomarkers(
            EGFR=None,
            ALK=None,
            PD_L1=None
        ),
        disallowed_conditions=["kidney disease"]
    )
    
    eligible = is_eligible(patient_high_creatinine, trial_3)
    inclusion_pass = check_inclusion(patient_high_creatinine, trial_3)
    exclusion_triggered = check_exclusion(patient_high_creatinine, trial_3)
    
    print(f"\nPatient: {patient_high_creatinine.age}yo {patient_high_creatinine.gender}, {patient_high_creatinine.cancer_type}, stage {patient_high_creatinine.stage}")
    print(f"  Lab values: hb={patient_high_creatinine.lab_values.hb}, wbc={patient_high_creatinine.lab_values.wbc}, creatinine={patient_high_creatinine.lab_values.creatinine}")
    print(f"  Comorbidities: {patient_high_creatinine.comorbidities}")
    print(f"\nTrial: {trial_3.trial_id}")
    print(f"  Inclusion: age 21-75, colon cancer, hb >= 9.0")
    print(f"  Exclusion: creatinine > 2.5, stage IV")
    print(f"  Disallowed: kidney disease")
    print(f"\n❌ Eligible: {eligible} (Exclusion Triggered)")
    print(f"   Reason: Patient creatinine {patient_high_creatinine.lab_values.creatinine} > threshold 2.5")
    print(f"   (Inclusion passed: {inclusion_pass}, Exclusion triggered: {exclusion_triggered})")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Case 1: ✅ Eligible (all criteria met)")
    print("Case 2: ❌ Not Eligible (inclusion failed - age too young)")
    print("Case 3: ❌ Not Eligible (exclusion triggered - high creatinine)")
    print()


if __name__ == "__main__":
    main()
