"""
Deterministic Eligibility Engine for Clinical Trial Matching

Pure logic implementation with NO randomness, NO LLM.
Evaluates patient eligibility against trial criteria.
"""

from src.schemas.patient_schema import Patient
from src.schemas.trial_schema import ClinicalTrial, Rule
from typing import Any


def get_nested_value(obj: Any, field_path: str) -> Any:
    """
    Extract nested field value from object using dot notation.
    
    Args:
        obj: Object to extract value from (typically Patient)
        field_path: Dot-separated field path (e.g., "lab_values.hb")
    
    Returns:
        The value at the specified field path
    
    Raises:
        ValueError: If field path is invalid or field does not exist
    
    Examples:
        get_nested_value(patient, "age") -> 45
        get_nested_value(patient, "lab_values.hb") -> 12.5
        get_nested_value(patient, "biomarkers.EGFR") -> True
    """
    if not field_path:
        raise ValueError("field_path cannot be empty")
    
    parts = field_path.split(".")
    current = obj
    
    for i, part in enumerate(parts):
        if not hasattr(current, part):
            traversed = ".".join(parts[:i]) if i > 0 else "root"
            raise ValueError(
                f"Field '{part}' does not exist in {traversed}. "
                f"Full path: '{field_path}'"
            )
        current = getattr(current, part)
    
    return current


def evaluate_rule(patient: Patient, rule: Rule) -> bool:
    """
    Evaluate a single eligibility rule against a patient.
    
    Args:
        patient: Patient to evaluate
        rule: Rule to apply
    
    Returns:
        True if rule passes, False otherwise
    
    Raises:
        ValueError: If field path is invalid or type mismatch occurs
    
    Type Safety:
        - numeric vs numeric ONLY (int, float, NOT bool)
        - string vs string ONLY
        - bool vs bool ONLY (separate from numeric)
        - Categorical fields (stage, gender, cancer_type) only allow == or !=
        - Mismatches raise ValueError
    """
    patient_value = get_nested_value(patient, rule.field)
    rule_value = rule.value
    
    patient_type = type(patient_value)
    rule_type = type(rule_value)
    
    is_patient_bool = isinstance(patient_value, bool)
    is_rule_bool = isinstance(rule_value, bool)
    
    if is_patient_bool != is_rule_bool:
        raise ValueError(
            f"Bool/numeric type confusion: "
            f"patient field '{rule.field}' is {'bool' if is_patient_bool else 'numeric'} (value: {patient_value}), "
            f"but rule value is {'bool' if is_rule_bool else 'numeric'} (value: {rule_value}). "
            f"Bool must be compared with bool only."
        )
    
    is_patient_numeric = isinstance(patient_value, (int, float)) and not isinstance(patient_value, bool)
    is_rule_numeric = isinstance(rule_value, (int, float)) and not isinstance(rule_value, bool)
    is_patient_str = isinstance(patient_value, str)
    is_rule_str = isinstance(rule_value, str)
    
    if is_patient_numeric and is_rule_numeric:
        pass
    elif is_patient_bool and is_rule_bool:
        pass
    elif is_patient_str and is_rule_str:
        pass
    else:
        raise ValueError(
            f"Type mismatch in rule evaluation: "
            f"patient field '{rule.field}' has type {patient_type.__name__} (value: {patient_value}), "
            f"but rule expects type {rule_type.__name__} (value: {rule_value})"
        )
    
    categorical_fields = {"stage", "gender", "cancer_type"}
    if rule.field in categorical_fields:
        if rule.operator not in {"==", "!="}:
            raise ValueError(
                f"Categorical field '{rule.field}' cannot use numeric operator '{rule.operator}'. "
                f"Only '==' and '!=' are allowed for categorical fields."
            )
    
    if rule.operator == ">":
        if is_patient_str:
            raise ValueError(
                f"Operator '>' cannot be used with string field '{rule.field}'. "
                f"Use '==' or '!=' for string comparisons."
            )
        return patient_value > rule_value
    elif rule.operator == "<":
        if is_patient_str:
            raise ValueError(
                f"Operator '<' cannot be used with string field '{rule.field}'. "
                f"Use '==' or '!=' for string comparisons."
            )
        return patient_value < rule_value
    elif rule.operator == ">=":
        if is_patient_str:
            raise ValueError(
                f"Operator '>=' cannot be used with string field '{rule.field}'. "
                f"Use '==' or '!=' for string comparisons."
            )
        return patient_value >= rule_value
    elif rule.operator == "<=":
        if is_patient_str:
            raise ValueError(
                f"Operator '<=' cannot be used with string field '{rule.field}'. "
                f"Use '==' or '!=' for string comparisons."
            )
        return patient_value <= rule_value
    elif rule.operator == "==":
        return patient_value == rule_value
    elif rule.operator == "!=":
        return patient_value != rule_value
    else:
        raise ValueError(f"Unknown operator: {rule.operator}")


def check_inclusion(patient: Patient, trial: ClinicalTrial) -> bool:
    """
    Check if patient passes ALL inclusion criteria.
    
    Args:
        patient: Patient to evaluate
        trial: Clinical trial with inclusion criteria
    
    Returns:
        True if ALL inclusion rules pass, False otherwise
    """
    for rule in trial.inclusion_criteria:
        if not evaluate_rule(patient, rule):
            return False
    return True


def check_exclusion(patient: Patient, trial: ClinicalTrial) -> bool:
    """
    Check if patient triggers ANY exclusion criteria.
    
    Args:
        patient: Patient to evaluate
        trial: Clinical trial with exclusion criteria
    
    Returns:
        True if ANY exclusion rule is triggered, False otherwise
    """
    for rule in trial.exclusion_criteria:
        if evaluate_rule(patient, rule):
            return True
    return False


def check_biomarkers(patient: Patient, trial: ClinicalTrial) -> bool:
    """
    Check if patient's biomarkers meet trial requirements.
    
    Rules:
        - If trial biomarker is None → ignore (not required)
        - If trial biomarker is bool → must match exactly
        - If trial PD_L1 is float → patient.PD_L1 >= trial.PD_L1 (threshold)
    
    Args:
        patient: Patient to evaluate
        trial: Clinical trial with biomarker requirements
    
    Returns:
        True if all biomarker requirements are met, False otherwise
    """
    trial_biomarkers = trial.required_biomarkers
    patient_biomarkers = patient.biomarkers
    
    if trial_biomarkers.EGFR is not None:
        if patient_biomarkers.EGFR != trial_biomarkers.EGFR:
            return False
    
    if trial_biomarkers.ALK is not None:
        if patient_biomarkers.ALK != trial_biomarkers.ALK:
            return False
    
    if trial_biomarkers.PD_L1 is not None:
        if patient_biomarkers.PD_L1 < trial_biomarkers.PD_L1:
            return False
    
    if trial_biomarkers.EGFR_expression_min is not None:
        if patient_biomarkers.EGFR_expression < trial_biomarkers.EGFR_expression_min:
            return False
    
    if trial_biomarkers.ALK_expression_min is not None:
        if patient_biomarkers.ALK_expression < trial_biomarkers.ALK_expression_min:
            return False
    
    return True


SEVERITY_ORDER = {"mild": 1, "moderate": 2, "severe": 3}


def check_comorbidities(patient: Patient, trial: ClinicalTrial) -> bool:
    """
    Check if patient has any disallowed comorbidities at or above threshold severity.
    
    Args:
        patient: Patient to evaluate
        trial: Clinical trial with disallowed conditions
    
    Returns:
        True if patient has NO disallowed comorbidities at threshold severity, False otherwise
    """
    for disallowed in trial.disallowed_conditions:
        for comorbidity in patient.comorbidities:
            if comorbidity.name == disallowed.name:
                patient_sev = SEVERITY_ORDER[comorbidity.severity]
                min_sev = SEVERITY_ORDER[disallowed.min_severity]
                if patient_sev >= min_sev:
                    return False
    return True


def evaluate_rule_with_detail(patient: Patient, rule: Rule) -> dict:
    """
    Evaluate a single rule and return detailed result.
    Returns:
    {
        "field": "age",
        "operator": ">=",
        "value": 18,
        "patient_value": 44,
        "passed": True,
        "explanation": "age 44 >= 18 ✓"
    }
    """
    try:
        patient_value = get_nested_value(patient, rule.field)
        passed = evaluate_rule(patient, rule)

        if passed:
            explanation = f"{rule.field} {patient_value} {rule.operator} {rule.value} ✓"
        else:
            explanation = f"{rule.field} {patient_value} {rule.operator} {rule.value} ✗"

        return {
            "field": rule.field,
            "operator": rule.operator,
            "value": rule.value,
            "patient_value": patient_value,
            "passed": passed,
            "explanation": explanation
        }
    except Exception as e:
        return {
            "field": rule.field,
            "operator": rule.operator,
            "value": rule.value,
            "patient_value": None,
            "passed": False,
            "explanation": f"Error: {str(e)}"
        }


def get_eligibility_details(patient: Patient, trial: ClinicalTrial) -> dict:
    """
    Returns full detailed eligibility breakdown.
    """
    inclusion_details = []
    for rule in trial.inclusion_criteria:
        detail = evaluate_rule_with_detail(patient, rule)
        inclusion_details.append(detail)

    exclusion_details = []
    for rule in trial.exclusion_criteria:
        detail = evaluate_rule_with_detail(patient, rule)
        exclusion_details.append(detail)

    inclusion_pass = all(d["passed"] for d in inclusion_details)
    exclusion_triggered = any(d["passed"] for d in exclusion_details)

    biomarker_details = {}
    bm = trial.required_biomarkers

    if bm.EGFR is not None:
        patient_egfr = patient.biomarkers.EGFR
        passed = patient_egfr == bm.EGFR
        biomarker_details["EGFR"] = {
            "required": bm.EGFR,
            "patient_value": patient_egfr,
            "passed": passed,
            "explanation": f"EGFR {patient_egfr} == {bm.EGFR} {'✓' if passed else '✗'}"
        }
    else:
        biomarker_details["EGFR"] = {
            "required": None,
            "patient_value": patient.biomarkers.EGFR,
            "passed": True,
            "explanation": "EGFR not required ✓"
        }

    if bm.ALK is not None:
        patient_alk = patient.biomarkers.ALK
        passed = patient_alk == bm.ALK
        biomarker_details["ALK"] = {
            "required": bm.ALK,
            "patient_value": patient_alk,
            "passed": passed,
            "explanation": f"ALK {patient_alk} == {bm.ALK} {'✓' if passed else '✗'}"
        }
    else:
        biomarker_details["ALK"] = {
            "required": None,
            "patient_value": patient.biomarkers.ALK,
            "passed": True,
            "explanation": "ALK not required ✓"
        }

    if bm.PD_L1 is not None:
        patient_pdl1 = patient.biomarkers.PD_L1
        passed = patient_pdl1 >= bm.PD_L1
        biomarker_details["PD_L1"] = {
            "required": f">= {bm.PD_L1}",
            "patient_value": patient_pdl1,
            "passed": passed,
            "explanation": f"PD_L1 {patient_pdl1} >= {bm.PD_L1} {'✓' if passed else '✗'}"
        }
    else:
        biomarker_details["PD_L1"] = {
            "required": None,
            "patient_value": patient.biomarkers.PD_L1,
            "passed": True,
            "explanation": "PD_L1 not required ✓"
        }

    biomarkers_pass = all(
        d["passed"] for d in biomarker_details.values()
    )

    conflicts = []
    for disallowed in trial.disallowed_conditions:
        for comorbidity in patient.comorbidities:
            if comorbidity.name == disallowed.name:
                patient_sev = SEVERITY_ORDER[comorbidity.severity]
                min_sev = SEVERITY_ORDER[disallowed.min_severity]
                if patient_sev >= min_sev:
                    conflicts.append(f"{comorbidity.name} ({comorbidity.severity} >= {disallowed.min_severity})")
    comorbidities_pass = len(conflicts) == 0
    comorbidity_details = {
        "disallowed": [{"name": d.name, "min_severity": d.min_severity} for d in trial.disallowed_conditions],
        "patient_has": [{"name": c.name, "severity": c.severity} for c in patient.comorbidities],
        "conflicts": conflicts,
        "passed": comorbidities_pass
    }

    prior_treatment_pass = check_prior_treatments(patient, trial)
    prior_treatment_details = {
        "required": trial.required_prior_treatments,
        "forbidden": trial.forbidden_prior_treatments,
        "patient_treatments": patient.prior_treatments,
        "missing_required": [
            t for t in trial.required_prior_treatments
            if t not in patient.prior_treatments
        ],
        "has_forbidden": [
            t for t in trial.forbidden_prior_treatments
            if t in patient.prior_treatments
        ],
        "passed": prior_treatment_pass
    }

    has_capacity = trial.has_capacity
    capacity_details = {
        "max_patients": trial.max_patients,
        "enrolled_patients": trial.enrolled_patients,
        "spots_remaining": trial.spots_remaining,
        "has_capacity": has_capacity
    }

    eligible = (
        has_capacity and
        inclusion_pass and
        not exclusion_triggered and
        biomarkers_pass and
        comorbidities_pass and
        prior_treatment_pass
    )

    if eligible:
        summary = "Patient ELIGIBLE for this trial ✓"
    else:
        reasons = []
        if not has_capacity:
            reasons.append(f"No capacity (enrolled {trial.enrolled_patients}/{trial.max_patients})")
        if not inclusion_pass:
            failed = [d["explanation"] for d in inclusion_details
                     if not d["passed"]]
            reasons.append(f"Inclusion failed: {'; '.join(failed)}")
        if exclusion_triggered:
            triggered = [d["explanation"] for d in exclusion_details
                        if d["passed"]]
            reasons.append(f"Exclusion triggered: {'; '.join(triggered)}")
        if not biomarkers_pass:
            failed_bm = [v["explanation"] for v in biomarker_details.values()
                        if not v["passed"]]
            reasons.append(f"Biomarkers: {'; '.join(failed_bm)}")
        if not comorbidities_pass:
            reasons.append(f"Comorbidities conflict: {conflicts}")
        if not prior_treatment_pass:
            missing = prior_treatment_details["missing_required"]
            has_forbidden = prior_treatment_details["has_forbidden"]
            parts = []
            if missing:
                parts.append(f"missing required: {missing}")
            if has_forbidden:
                parts.append(f"has forbidden: {has_forbidden}")
            reasons.append(f"Prior treatments: {'; '.join(parts)}")
        summary = "Patient INELIGIBLE: " + " | ".join(reasons)

    return {
        "eligible": eligible,
        "inclusion_pass": inclusion_pass,
        "exclusion_triggered": exclusion_triggered,
        "biomarkers_pass": biomarkers_pass,
        "comorbidities_pass": comorbidities_pass,
        "prior_treatment_pass": prior_treatment_pass,
        "has_capacity": has_capacity,
        "inclusion_details": inclusion_details,
        "exclusion_details": exclusion_details,
        "biomarker_details": biomarker_details,
        "comorbidity_details": comorbidity_details,
        "prior_treatment_details": prior_treatment_details,
        "capacity_details": capacity_details,
        "summary": summary
    }


def check_prior_treatments(patient: Patient, trial: ClinicalTrial) -> bool:
    """
    Check if patient's prior treatments meet trial requirements.
    
    Args:
        patient: Patient to evaluate
        trial: Clinical trial with treatment requirements
    
    Returns:
        True if patient meets all treatment requirements, False otherwise
    """
    for required in trial.required_prior_treatments:
        if required not in patient.prior_treatments:
            return False
    for forbidden in trial.forbidden_prior_treatments:
        if forbidden in patient.prior_treatments:
            return False
    return True


def is_eligible(patient: Patient, trial: ClinicalTrial) -> bool:
    """
    Determine if a patient is eligible for a clinical trial.
    
    Eligibility requires:
        - Trial has capacity
        - ALL inclusion rules pass
        - NO exclusion rules triggered
        - Biomarkers match requirements
        - No disallowed comorbidities
        - Prior treatment requirements met
    
    Args:
        patient: Patient to evaluate
        trial: Clinical trial to check eligibility for
    
    Returns:
        True if patient is eligible, False otherwise
    """
    if not trial.has_capacity:
        return False
    
    inclusion_pass = check_inclusion(patient, trial)
    exclusion_triggered = check_exclusion(patient, trial)
    biomarker_pass = check_biomarkers(patient, trial)
    comorbidity_pass = check_comorbidities(patient, trial)
    prior_treatment_pass = check_prior_treatments(patient, trial)
    
    return (inclusion_pass and not exclusion_triggered and biomarker_pass
            and comorbidity_pass and prior_treatment_pass)
