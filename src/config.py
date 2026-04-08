"""
Global Configuration for ClinicalTrialMatchEnv

Supports TWO MODES:
1. STRICT_MODE: Full compliance with Phase 1 spec (for evaluation/judging)
2. REALISTIC_MODE: Advanced simulation with missing data, conflicts, trends

Only ONE mode can be active at a time.
"""

# MODE SELECTION
# NOTE: Tasks were designed with REALISTIC_MODE
# Use STRICT_MODE only for Phase 1 spec compliance testing
STRICT_MODE = False
REALISTIC_MODE = True


def validate_mode():
    """Ensure only one mode is active."""
    if STRICT_MODE and REALISTIC_MODE:
        raise ValueError(
            "Configuration Error: Both STRICT_MODE and REALISTIC_MODE are True. "
            "Only one mode can be active at a time."
        )
    
    if not STRICT_MODE and not REALISTIC_MODE:
        raise ValueError(
            "Configuration Error: Both STRICT_MODE and REALISTIC_MODE are False. "
            "At least one mode must be active."
        )


# Validate on import
validate_mode()


def get_active_mode() -> str:
    """Return the name of the active mode."""
    if STRICT_MODE:
        return "STRICT_MODE"
    elif REALISTIC_MODE:
        return "REALISTIC_MODE"
    else:
        raise ValueError("No mode is active")


# Grader score clamping (for hackathon submission only)
# Set to True for hackathon (scores must be in (0,1) not [0,1])
# Set to False for internal testing (allows exact 0.0 and 1.0)
CLAMP_SCORES_FOR_HACKATHON = False

# Mode-specific settings
if STRICT_MODE:
    # STRICT_MODE: 100% spec compliance
    ALLOW_MISSING_LAB_VALUES = False
    ALLOW_CONFLICTING_FIELDS = False
    ALLOW_TEMPORAL_TRENDS = False
    CANCER_TYPE_STRICT_ENUM = True
    
elif REALISTIC_MODE:
    # REALISTIC_MODE: Advanced simulation
    ALLOW_MISSING_LAB_VALUES = True
    ALLOW_CONFLICTING_FIELDS = True
    ALLOW_TEMPORAL_TRENDS = True
    CANCER_TYPE_STRICT_ENUM = False
    
    # Probabilities for realistic features
    MISSING_LAB_VALUE_PROBABILITY = 0.15  # 15% chance each lab is unknown
    CONFLICTING_FIELD_PROBABILITY = 0.20  # 20% chance of data conflicts
