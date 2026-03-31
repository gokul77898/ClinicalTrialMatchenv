"""Eligibility engine for clinical trial matching."""

from .eligibility_engine import (
    get_nested_value,
    evaluate_rule,
    check_inclusion,
    check_exclusion,
    check_biomarkers,
    check_comorbidities,
    is_eligible,
)

__all__ = [
    'get_nested_value',
    'evaluate_rule',
    'check_inclusion',
    'check_exclusion',
    'check_biomarkers',
    'check_comorbidities',
    'is_eligible',
]
