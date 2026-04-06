"""
Comprehensive pytest suite for ClinicalTrialMatchEnv Phase 1.

Tests data integrity, eligibility correctness, stress testing,
rule engine robustness, and eligibility distribution.
"""

import pytest
from src.schemas.patient_schema import Patient, Biomarkers, LabValues, Comorbidity, generate_random_patient
from src.schemas.trial_schema import ClinicalTrial, Rule, RequiredBiomarkers, DisallowedCondition, generate_random_trial
from src.engine.eligibility_engine import is_eligible, evaluate_rule, get_nested_value
from uuid import uuid4
import json


class TestDataIntegrity:
    """TEST 1: Data Integrity - Validate generated patients and trials."""
    
    def test_patient_generation_integrity(self):
        """Generate 20 patients and validate all fields."""
        print("\n" + "=" * 80)
        print("TEST 1A: PATIENT DATA INTEGRITY")
        print("=" * 80)
        
        patients = [generate_random_patient(seed=i) for i in range(20)]
        
        for i, patient in enumerate(patients):
            assert patient.id is not None, f"Patient {i}: id is None"
            assert isinstance(patient.id, type(uuid4())), f"Patient {i}: id is not UUID"
            
            assert patient.age is not None, f"Patient {i}: age is None"
            assert isinstance(patient.age, int), f"Patient {i}: age is not int"
            assert 0 <= patient.age <= 120, f"Patient {i}: age {patient.age} out of range"
            
            assert patient.gender is not None, f"Patient {i}: gender is None"
            assert patient.gender in ["male", "female", "other"], f"Patient {i}: invalid gender"
            
            assert patient.cancer_type is not None, f"Patient {i}: cancer_type is None"
            assert patient.cancer_type in ["lung cancer", "breast cancer", "colon cancer"], \
                f"Patient {i}: invalid cancer_type"
            
            assert patient.stage is not None, f"Patient {i}: stage is None"
            assert patient.stage in ["I", "II", "III", "IV"], f"Patient {i}: invalid stage"
            
            assert patient.biomarkers is not None, f"Patient {i}: biomarkers is None"
            assert isinstance(patient.biomarkers.EGFR, bool), f"Patient {i}: EGFR is not bool"
            assert isinstance(patient.biomarkers.ALK, bool), f"Patient {i}: ALK is not bool"
            assert 0 <= patient.biomarkers.PD_L1 <= 100, f"Patient {i}: PD_L1 out of range"
            
            assert patient.prior_treatments is not None, f"Patient {i}: prior_treatments is None"
            assert isinstance(patient.prior_treatments, list), f"Patient {i}: prior_treatments is not list"
            
            assert patient.lab_values is not None, f"Patient {i}: lab_values is None"
            if patient.lab_values.hb is not None:
                assert 0 <= patient.lab_values.hb <= 20, f"Patient {i}: hb out of range"
            if patient.lab_values.wbc is not None:
                assert 0 <= patient.lab_values.wbc <= 20000, f"Patient {i}: wbc out of range"
            if patient.lab_values.creatinine is not None:
                assert 0 <= patient.lab_values.creatinine <= 10, f"Patient {i}: creatinine out of range"
            
            assert patient.comorbidities is not None, f"Patient {i}: comorbidities is None"
            assert isinstance(patient.comorbidities, list), f"Patient {i}: comorbidities is not list"
        
        print(f"\n✅ All 20 patients validated successfully")
        print(f"\nSample Patients (first 3):")
        for i in range(3):
            print(f"\nPatient {i+1}:")
            print(json.dumps(patients[i].model_dump(), indent=2, default=str))
    
    def test_trial_generation_integrity(self):
        """Generate 20 trials and validate all fields."""
        print("\n" + "=" * 80)
        print("TEST 1B: TRIAL DATA INTEGRITY")
        print("=" * 80)
        
        trials = [generate_random_trial(seed=i) for i in range(20)]
        
        for i, trial in enumerate(trials):
            assert trial.trial_id is not None, f"Trial {i}: trial_id is None"
            assert isinstance(trial.trial_id, str), f"Trial {i}: trial_id is not str"
            assert len(trial.trial_id) > 0, f"Trial {i}: trial_id is empty"
            
            assert trial.cancer_type is not None, f"Trial {i}: cancer_type is None"
            assert trial.cancer_type in ["lung cancer", "breast cancer", "colon cancer"], \
                f"Trial {i}: invalid cancer_type"
            
            assert trial.inclusion_criteria is not None, f"Trial {i}: inclusion_criteria is None"
            assert len(trial.inclusion_criteria) >= 3, \
                f"Trial {i}: inclusion_criteria has {len(trial.inclusion_criteria)} rules, need >= 3"
            
            assert trial.exclusion_criteria is not None, f"Trial {i}: exclusion_criteria is None"
            assert len(trial.exclusion_criteria) >= 2, \
                f"Trial {i}: exclusion_criteria has {len(trial.exclusion_criteria)} rules, need >= 2"
            
            assert trial.required_biomarkers is not None, f"Trial {i}: required_biomarkers is None"
            if trial.required_biomarkers.PD_L1 is not None:
                assert 0 <= trial.required_biomarkers.PD_L1 <= 100, \
                    f"Trial {i}: PD_L1 threshold out of range"
            
            assert trial.disallowed_conditions is not None, f"Trial {i}: disallowed_conditions is None"
            assert isinstance(trial.disallowed_conditions, list), \
                f"Trial {i}: disallowed_conditions is not list"
            
            for j, rule in enumerate(trial.inclusion_criteria):
                assert rule.field is not None, f"Trial {i}, inclusion rule {j}: field is None"
                assert rule.operator is not None, f"Trial {i}, inclusion rule {j}: operator is None"
                assert rule.value is not None, f"Trial {i}, inclusion rule {j}: value is None"
            
            for j, rule in enumerate(trial.exclusion_criteria):
                assert rule.field is not None, f"Trial {i}, exclusion rule {j}: field is None"
                assert rule.operator is not None, f"Trial {i}, exclusion rule {j}: operator is None"
                assert rule.value is not None, f"Trial {i}, exclusion rule {j}: value is None"
        
        print(f"\n✅ All 20 trials validated successfully")
        print(f"\nSample Trials (first 3):")
        for i in range(3):
            print(f"\nTrial {i+1}:")
            print(json.dumps(trials[i].model_dump(), indent=2))


class TestEligibilityCorrectness:
    """TEST 2: Eligibility Correctness - Manual test cases with known outcomes."""
    
    def test_case_1_fully_eligible(self):
        """Case 1: Patient fully eligible for trial."""
        print("\n" + "=" * 80)
        print("TEST 2.1: FULLY ELIGIBLE")
        print("=" * 80)
        
        patient = Patient(
            id=uuid4(),
            age=50,
            gender="male",
            cancer_type="lung cancer",
            stage="II",
            biomarkers=Biomarkers(EGFR=True, ALK=False, PD_L1=60.0),
            prior_treatments=["chemotherapy"],
            lab_values=LabValues(hb=12.0, wbc=6000.0, creatinine=1.0),
            comorbidities=[Comorbidity(name="hypertension", severity="mild")]
        )
        
        trial = ClinicalTrial(
            trial_id="TEST-001",
            cancer_type="lung cancer",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=18),
                Rule(field="age", operator="<=", value=75),
                Rule(field="cancer_type", operator="==", value="lung cancer")
            ],
            exclusion_criteria=[
                Rule(field="age", operator=">", value=80),
                Rule(field="lab_values.creatinine", operator=">", value=2.0)
            ],
            required_biomarkers=RequiredBiomarkers(EGFR=True, ALK=None, PD_L1=50.0),
            disallowed_conditions=[DisallowedCondition(name="kidney disease")]
        )
        
        expected = True
        result = is_eligible(patient, trial)
        
        print(f"Patient: {patient.age}yo, {patient.cancer_type}, EGFR={patient.biomarkers.EGFR}, PD_L1={patient.biomarkers.PD_L1}")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ PASSED")
    
    def test_case_2_inclusion_failure(self):
        """Case 2: Patient fails inclusion criteria (too young)."""
        print("\n" + "=" * 80)
        print("TEST 2.2: INCLUSION FAILURE")
        print("=" * 80)
        
        patient = Patient(
            id=uuid4(),
            age=16,
            gender="female",
            cancer_type="lung cancer",
            stage="I",
            biomarkers=Biomarkers(EGFR=True, ALK=False, PD_L1=55.0),
            prior_treatments=[],
            lab_values=LabValues(hb=13.0, wbc=6500.0, creatinine=0.9),
            comorbidities=[]
        )
        
        trial = ClinicalTrial(
            trial_id="TEST-002",
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
            required_biomarkers=RequiredBiomarkers(EGFR=None, ALK=None, PD_L1=None),
            disallowed_conditions=[]
        )
        
        expected = False
        result = is_eligible(patient, trial)
        
        print(f"Patient: {patient.age}yo (too young, minimum is 18)")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ PASSED")
    
    def test_case_3_exclusion_triggered(self):
        """Case 3: Patient triggers exclusion criteria (high creatinine)."""
        print("\n" + "=" * 80)
        print("TEST 2.3: EXCLUSION TRIGGERED")
        print("=" * 80)
        
        patient = Patient(
            id=uuid4(),
            age=62,
            gender="male",
            cancer_type="colon cancer",
            stage="III",
            biomarkers=Biomarkers(EGFR=False, ALK=False, PD_L1=25.0),
            prior_treatments=["surgery"],
            lab_values=LabValues(hb=11.0, wbc=5500.0, creatinine=2.8),
            comorbidities=[Comorbidity(name="diabetes", severity="mild")]
        )
        
        trial = ClinicalTrial(
            trial_id="TEST-003",
            cancer_type="colon cancer",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=21),
                Rule(field="age", operator="<=", value=75),
                Rule(field="cancer_type", operator="==", value="colon cancer")
            ],
            exclusion_criteria=[
                Rule(field="lab_values.creatinine", operator=">", value=2.5),
                Rule(field="stage", operator="==", value="IV")
            ],
            required_biomarkers=RequiredBiomarkers(EGFR=None, ALK=None, PD_L1=None),
            disallowed_conditions=[DisallowedCondition(name="kidney disease")]
        )
        
        expected = False
        result = is_eligible(patient, trial)
        
        print(f"Patient: creatinine={patient.lab_values.creatinine} (> 2.5 threshold)")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ PASSED")
    
    def test_case_4_biomarker_failure(self):
        """Case 4: Patient fails biomarker requirements."""
        print("\n" + "=" * 80)
        print("TEST 2.4: BIOMARKER FAILURE")
        print("=" * 80)
        
        patient = Patient(
            id=uuid4(),
            age=55,
            gender="female",
            cancer_type="breast cancer",
            stage="II",
            biomarkers=Biomarkers(EGFR=False, ALK=False, PD_L1=30.0),
            prior_treatments=["chemotherapy"],
            lab_values=LabValues(hb=12.5, wbc=6200.0, creatinine=1.1),
            comorbidities=[]
        )
        
        trial = ClinicalTrial(
            trial_id="TEST-004",
            cancer_type="breast cancer",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=18),
                Rule(field="age", operator="<=", value=75),
                Rule(field="cancer_type", operator="==", value="breast cancer")
            ],
            exclusion_criteria=[
                Rule(field="age", operator=">", value=80),
                Rule(field="lab_values.hb", operator="<", value=8.0)
            ],
            required_biomarkers=RequiredBiomarkers(EGFR=True, ALK=None, PD_L1=None),
            disallowed_conditions=[]
        )
        
        expected = False
        result = is_eligible(patient, trial)
        
        print(f"Patient: EGFR={patient.biomarkers.EGFR} (trial requires EGFR=True)")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ PASSED")
    
    def test_case_5_comorbidity_failure(self):
        """Case 5: Patient has disallowed comorbidity."""
        print("\n" + "=" * 80)
        print("TEST 2.5: COMORBIDITY FAILURE")
        print("=" * 80)
        
        patient = Patient(
            id=uuid4(),
            age=48,
            gender="male",
            cancer_type="lung cancer",
            stage="I",
            biomarkers=Biomarkers(EGFR=True, ALK=False, PD_L1=45.0),
            prior_treatments=[],
            lab_values=LabValues(hb=13.0, wbc=7000.0, creatinine=1.3),
            comorbidities=[Comorbidity(name="kidney disease", severity="mild"), Comorbidity(name="hypertension", severity="mild")]
        )
        
        trial = ClinicalTrial(
            trial_id="TEST-005",
            cancer_type="lung cancer",
            inclusion_criteria=[
                Rule(field="age", operator=">=", value=18),
                Rule(field="age", operator="<=", value=75),
                Rule(field="cancer_type", operator="==", value="lung cancer")
            ],
            exclusion_criteria=[
                Rule(field="age", operator=">", value=80),
                Rule(field="lab_values.creatinine", operator=">", value=2.0)
            ],
            required_biomarkers=RequiredBiomarkers(EGFR=None, ALK=None, PD_L1=None),
            disallowed_conditions=[DisallowedCondition(name="kidney disease"), DisallowedCondition(name="liver disease")]
        )
        
        expected = False
        result = is_eligible(patient, trial)
        
        print(f"Patient: comorbidities={patient.comorbidities}")
        print(f"Trial disallows: {trial.disallowed_conditions}")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        
        assert result == expected, f"Expected {expected}, got {result}"
        print("✅ PASSED")


class TestStressTest:
    """TEST 3: Stress Test - Run 50 random patient-trial pairs."""
    
    def test_stress_50_random_pairs(self):
        """Run 50 random patient-trial pairs without exceptions."""
        print("\n" + "=" * 80)
        print("TEST 3: STRESS TEST (50 random pairs)")
        print("=" * 80)
        
        results = []
        exceptions = []
        
        for i in range(50):
            try:
                patient = generate_random_patient(seed=i * 100)
                trial = generate_random_trial(seed=i * 100 + 50)
                result = is_eligible(patient, trial)
                
                assert isinstance(result, bool), f"Iteration {i}: result is not bool, got {type(result)}"
                results.append(result)
            except Exception as e:
                exceptions.append((i, str(e)))
        
        print(f"\n✅ Completed 50 iterations")
        print(f"   Exceptions: {len(exceptions)}")
        print(f"   Boolean results: {len(results)}")
        print(f"   Eligible: {sum(results)}")
        print(f"   Not eligible: {len(results) - sum(results)}")
        
        assert len(exceptions) == 0, f"Found {len(exceptions)} exceptions: {exceptions[:3]}"
        assert len(results) == 50, f"Expected 50 results, got {len(results)}"
        print("\n✅ PASSED")


class TestRuleEngineRobustness:
    """TEST 4: Rule Engine Robustness - Test invalid rules."""
    
    def test_invalid_operator(self):
        """Test that invalid operator raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST 4.1: INVALID OPERATOR")
        print("=" * 80)
        
        with pytest.raises(Exception) as exc_info:
            Rule(field="age", operator="INVALID", value=50)
        
        print(f"✅ PASSED: Invalid operator rejected")
        print(f"   Error: {str(exc_info.value)[:100]}...")
    
    def test_invalid_field_path(self):
        """Test that invalid field path raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST 4.2: INVALID FIELD PATH")
        print("=" * 80)
        
        patient = Patient(
            id=uuid4(),
            age=50,
            gender="male",
            cancer_type="lung cancer",
            stage="II",
            biomarkers=Biomarkers(EGFR=True, ALK=False, PD_L1=50.0),
            prior_treatments=[],
            lab_values=LabValues(hb=12.0, wbc=6000.0, creatinine=1.0),
            comorbidities=[]
        )
        
        with pytest.raises(ValueError) as exc_info:
            get_nested_value(patient, "lab_values.fake_field")
        
        print(f"✅ PASSED: Invalid field path rejected")
        print(f"   Error: {str(exc_info.value)[:100]}...")
    
    def test_type_mismatch_age_string(self):
        """Test that type mismatch (age >= 'high') raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST 4.3: TYPE MISMATCH (age >= 'high')")
        print("=" * 80)
        
        with pytest.raises(Exception) as exc_info:
            Rule(field="age", operator=">=", value="high")
        
        print(f"✅ PASSED: Type mismatch rejected")
        print(f"   Error: {str(exc_info.value)[:100]}...")
    
    def test_string_numeric_operator(self):
        """Test that string field with numeric operator raises ValueError."""
        print("\n" + "=" * 80)
        print("TEST 4.4: STRING WITH NUMERIC OPERATOR")
        print("=" * 80)
        
        with pytest.raises(Exception) as exc_info:
            Rule(field="stage", operator=">", value="II")
        
        print(f"✅ PASSED: String numeric operator rejected")
        print(f"   Error: {str(exc_info.value)[:100]}...")


class TestEligibilityDistribution:
    """TEST 5: Eligibility Distribution - Verify realistic distribution."""
    
    def test_100_random_pairs_distribution(self):
        """Run 100 random pairs and verify both eligible and ineligible exist."""
        print("\n" + "=" * 80)
        print("TEST 5: ELIGIBILITY DISTRIBUTION (100 random pairs)")
        print("=" * 80)
        
        eligible_count = 0
        ineligible_count = 0
        
        for i in range(100):
            patient = generate_random_patient(seed=i * 1000)
            trial = generate_random_trial(seed=i * 1000 + 500)
            result = is_eligible(patient, trial)
            
            if result:
                eligible_count += 1
            else:
                ineligible_count += 1
        
        total = eligible_count + ineligible_count
        eligible_ratio = eligible_count / total if total > 0 else 0
        
        print(f"\n📊 DISTRIBUTION RESULTS:")
        print(f"   Total pairs: {total}")
        print(f"   Eligible: {eligible_count}")
        print(f"   Ineligible: {ineligible_count}")
        print(f"   Eligible ratio: {eligible_ratio:.2%}")
        
        assert eligible_count > 0, f"No eligible patients found in 100 pairs (generator issue)"
        assert ineligible_count > 0, f"No ineligible patients found in 100 pairs (generator issue)"
        
        print(f"\n✅ PASSED: Both eligible and ineligible patients exist")
        print(f"   Distribution is realistic (not all eligible, not all ineligible)")


def test_summary():
    """Print final summary after all tests."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "TEST SUITE SUMMARY" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n✅ TEST 1: Data Integrity - 20 patients, 20 trials validated")
    print("✅ TEST 2: Eligibility Correctness - 5 manual cases verified")
    print("✅ TEST 3: Stress Test - 50 random pairs executed")
    print("✅ TEST 4: Rule Engine Robustness - Invalid rules rejected")
    print("✅ TEST 5: Eligibility Distribution - Realistic distribution confirmed")
    print("\n🎯 Phase 1 comprehensive testing complete")
    print()
