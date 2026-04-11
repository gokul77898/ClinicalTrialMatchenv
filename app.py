"""
ClinicalTrialMatchEnv — Gradio Demo UI

Two modes:
  1. Synthetic tasks — run the deterministic agent on env-generated episodes
  2. Realistic cases — evaluate hand-crafted clinical scenarios with full trace
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import gradio as gr
from src.environment import ClinicalTrialEnv
from src.agents.clinical_trial_agent import ClinicalTrialAgent
from src.realistic_cases import (
    list_cases, load_case, evaluate_case,
    fmt_patient, fmt_trials_evaluated, fmt_decision,
)

# -----------------------------------------------
# SYNTHETIC TASKS
# -----------------------------------------------

TASKS = [
    ("single_match", "Easy — 1 correct trial, obvious fakes"),
    ("hidden_exclusion", "Medium — exclusion traps that look like matches"),
    ("ambiguous_match", "Hard — 7 trials, 3 exclusion traps, 2 biomarker failures"),
]


def _fmt_patient_obs(obs) -> str:
    p = obs.patient
    bio = p.biomarkers
    labs = p.lab_values
    lines = [
        f"Age:          {p.age}",
        f"Gender:       {p.gender}",
        f"Cancer Type:  {p.cancer_type}",
        f"Stage:        {p.stage}",
        "",
        "Biomarkers:",
        f"  EGFR:   {bio.EGFR}",
        f"  ALK:    {bio.ALK}",
        f"  PD_L1:  {bio.PD_L1}",
    ]
    if hasattr(bio, "EGFR_expression"):
        lines.append(f"  EGFR expression:  {bio.EGFR_expression}")
    if hasattr(bio, "ALK_expression"):
        lines.append(f"  ALK expression:   {bio.ALK_expression}")
    lines += [
        "", "Lab Values:",
        f"  HB:         {labs.hb if labs.hb is not None else 'unknown'}",
        f"  WBC:        {labs.wbc if labs.wbc is not None else 'unknown'}",
        f"  Creatinine: {labs.creatinine if labs.creatinine is not None else 'unknown'}",
    ]
    if hasattr(p, "comorbidities") and p.comorbidities:
        lines += ["", "Comorbidities:"]
        for c in p.comorbidities:
            if isinstance(c, dict):
                lines.append(f"  - {c.get('name', c)} ({c.get('severity', '?')})")
            else:
                lines.append(f"  - {c}")
    return "\n".join(lines)


def _fmt_trials_obs(obs) -> str:
    lines = []
    for i, t in enumerate(obs.available_trials):
        lines.append(f"Trial {i+1}: {t['trial_id']}")
        lines.append(f"  Cancer type:  {t['cancer_type']}")
        if "trial_score" in t:
            lines.append(f"  Quality:      {t['trial_score']}")
        if "max_patients" in t:
            lines.append(f"  Capacity:     {t.get('enrolled_patients', 0)}/{t['max_patients']}")
        if "days_until_deadline" in t:
            lines.append(f"  Deadline:     {t['days_until_deadline']} days")
        lines.append("")
    return "\n".join(lines).strip()


def _fmt_decision_analysis(result: dict, obs) -> str:
    """Build decision analysis from agent trial_analysis data."""
    analysis = result.get("trial_analysis", {})
    selected = result.get("selected_trial")
    all_trial_ids = [t["trial_id"] for t in obs.available_trials]

    lines = []

    # Selected trial
    if selected and selected in analysis:
        a = analysis[selected]
        lines.append(f"✅ SELECTED: {selected}  (score {a['score']:.1f})")
        for r in a["reasons"]:
            lines.append(f"   • {r}")
    elif selected:
        lines.append(f"✅ SELECTED: {selected}  (fallback — no scored trials)")
    lines.append("")

    # Rejected / not checked
    lines.append("REJECTED / NOT CHECKED:")
    for tid in all_trial_ids:
        if tid == selected:
            continue
        if tid in analysis:
            a = analysis[tid]
            if a["exclusion_triggered"]:
                lines.append(f"   ✖ {tid}: Rejected — exclusion triggered")
            elif not a["inclusion_pass"]:
                lines.append(f"   ✖ {tid}: Rejected — inclusion failed")
            else:
                lines.append(f"   ○ {tid}: Eligible but lower score ({a['score']:.1f})")
            for r in a["reasons"]:
                lines.append(f"      {r}")
        else:
            # Trial was not checked (wrong cancer type or step limit)
            trial_data = next((t for t in obs.available_trials if t["trial_id"] == tid), {})
            cancer = trial_data.get("cancer_type", "?")
            if cancer != obs.patient.cancer_type:
                lines.append(f"   ⊘ {tid}: Skipped — cancer type mismatch ({cancer})")
            else:
                lines.append(f"   ⊘ {tid}: Not checked (step budget)")

    return "\n".join(lines)


def run_synthetic(task_choice: str) -> tuple:
    """Run agent on a synthetic env task."""
    task_id = task_choice.split(" — ")[0].strip()
    env = ClinicalTrialEnv()
    agent = ClinicalTrialAgent()

    obs = env.reset(task_id=task_id)
    patient_str = _fmt_patient_obs(obs)
    trials_str = _fmt_trials_obs(obs)

    time.sleep(0.3)  # small delay for UI realism
    result = agent.run_episode(env, task_id=task_id)

    steps_lines = []
    for i, step in enumerate(result.get("reasoning", []), 1):
        steps_lines.append(f"Step {i} → {step}")
    steps_str = "\n".join(steps_lines)

    status = "✅ CORRECT" if result["success"] else "❌ WRONG"
    result_str = (
        f"Selected Trial:  {result['selected_trial']}\n"
        f"Reward:          {result['reward']:+.2f}\n"
        f"Steps Used:      {result['steps']}\n"
        f"Result:          {status}"
    )

    decision_str = _fmt_decision_analysis(result, obs)

    return patient_str, trials_str, steps_str, result_str, decision_str


# -----------------------------------------------
# REALISTIC CASES
# -----------------------------------------------

def run_realistic(case_choice: str) -> tuple:
    """Evaluate a realistic case with full trace."""
    case_id = case_choice.split(" — ")[0].strip()
    case = load_case(case_id)
    if case is None:
        msg = f"Case {case_id} not found"
        return msg, msg, msg, msg

    time.sleep(0.3)  # small delay for UI realism

    result = evaluate_case(case)

    patient_str = fmt_patient(result["patient"])
    trials_str = fmt_trials_evaluated(result["trial_results"])
    decision_str = fmt_decision(result)

    # Build step trace showing evaluation order
    trace_lines = []
    for i, tr in enumerate(result["trial_results"], 1):
        tag = "✔" if (tr["inclusion_pass"] and not tr["exclusion_triggered"]) else "✖"
        trace_lines.append(f"Step {i} → check_criteria(\"{tr['trial_id']}\")  {tag}")
        if not tr["inclusion_pass"]:
            failed = [r for r in tr["inclusion_reasons"] if r.startswith("✖")]
            if failed:
                trace_lines.append(f"         {failed[0]}")
        elif tr["exclusion_triggered"]:
            exc = [r for r in tr["exclusion_reasons"] if r.startswith("EXCLUDED")]
            if exc:
                trace_lines.append(f"         {exc[0]}")
        else:
            trace_lines.append(f"         Eligible — score {tr['score']:.3f}")

    sel = result["selected"]
    if sel:
        trace_lines.append(f"\nStep {len(result['trial_results'])+1} → select_trial(\"{sel}\")")
        trace_lines.append(f"Step {len(result['trial_results'])+2} → resolve()")
    else:
        trace_lines.append(f"\nNo eligible trial — resolve()")

    steps_str = "\n".join(trace_lines)

    return patient_str, trials_str, steps_str, decision_str


# -----------------------------------------------
# GRADIO UI
# -----------------------------------------------

_real_cases = list_cases()
_real_choices = [f"{c['case_id']} — {c['label']}" for c in _real_cases]

with gr.Blocks(title="ClinicalTrialMatchEnv") as demo:
    gr.Markdown(
        "# 🏥 ClinicalTrialMatchEnv\n"
        "### AI agent for cancer patient → clinical trial matching\n"
        "Run **synthetic tasks** (env-generated) or **realistic cases** "
        "(hand-crafted clinical scenarios with full eligibility trace)."
    )

    with gr.Tabs():
        # --- Tab 1: Synthetic ---
        with gr.TabItem("Synthetic Tasks"):
            with gr.Row():
                syn_dropdown = gr.Dropdown(
                    choices=[f"{tid} — {desc}" for tid, desc in TASKS],
                    value=f"{TASKS[0][0]} — {TASKS[0][1]}",
                    label="Task Difficulty",
                )
                syn_btn = gr.Button("▶  Run Episode", variant="primary", scale=0)

            with gr.Row():
                with gr.Column():
                    syn_patient = gr.Textbox(label="Patient", lines=16, interactive=False)
                    syn_trials = gr.Textbox(label="Available Trials", lines=16, interactive=False)
                with gr.Column():
                    syn_steps = gr.Textbox(label="Agent Steps", lines=16, interactive=False)
                    syn_result = gr.Textbox(label="Final Result", lines=6, interactive=False)
                    syn_decision = gr.Textbox(
                        label="Decision Analysis — Why Selected / Why Rejected",
                        lines=12, interactive=False,
                    )

            syn_btn.click(
                fn=run_synthetic,
                inputs=[syn_dropdown],
                outputs=[syn_patient, syn_trials, syn_steps, syn_result, syn_decision],
            )

        # --- Tab 2: Realistic ---
        with gr.TabItem("Realistic Cases"):
            with gr.Row():
                real_dropdown = gr.Dropdown(
                    choices=_real_choices,
                    value=_real_choices[0] if _real_choices else None,
                    label="Case",
                )
                real_btn = gr.Button("▶  Evaluate Case", variant="primary", scale=0)

            with gr.Row():
                with gr.Column():
                    real_patient = gr.Textbox(label="Patient", lines=16, interactive=False)
                    real_trials = gr.Textbox(
                        label="Trials (with ✔/✖ + inclusion/exclusion trace)",
                        lines=20, interactive=False,
                    )
                with gr.Column():
                    real_steps = gr.Textbox(label="Decision Trace", lines=20, interactive=False)
                    real_decision = gr.Textbox(
                        label="Why Selected / Why Others Rejected",
                        lines=10, interactive=False,
                    )

            real_btn.click(
                fn=run_realistic,
                inputs=[real_dropdown],
                outputs=[real_patient, real_trials, real_steps, real_decision],
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
