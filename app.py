"""
ClinicalTrialMatchEnv — Gradio Demo UI

Allows users to run agent episodes and see step-by-step reasoning,
patient data, trial summaries, and final outcomes.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.environment import ClinicalTrialEnv
from src.agents.clinical_trial_agent import ClinicalTrialAgent

TASKS = [
    ("single_match", "Easy — 1 correct trial, obvious fakes"),
    ("hidden_exclusion", "Medium — exclusion traps that look like matches"),
    ("ambiguous_match", "Hard — 7 trials, 3 exclusion traps, 2 biomarker failures"),
]


def format_patient(obs) -> str:
    """Format patient data as readable text."""
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
        "",
        "Lab Values:",
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


def format_trials(obs) -> str:
    """Format available trials as readable text."""
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


def format_steps(reasoning: list, result: dict) -> str:
    """Format step-by-step agent reasoning."""
    lines = []
    for i, step in enumerate(reasoning, 1):
        lines.append(f"Step {i} → {step}")
    return "\n".join(lines)


def format_result(result: dict) -> str:
    """Format final outcome."""
    status = "✅ CORRECT" if result["success"] else "❌ WRONG"
    lines = [
        f"Selected Trial:  {result['selected_trial']}",
        f"Reward:          {result['reward']:+.2f}",
        f"Steps Used:      {result['steps']}",
        f"Result:          {status}",
    ]
    return "\n".join(lines)


def run_demo(task_choice: str) -> tuple:
    """Run one episode and return formatted outputs."""
    task_id = task_choice.split(" — ")[0].strip()

    env = ClinicalTrialEnv()
    agent = ClinicalTrialAgent()

    # Capture initial observation for display
    obs = env.reset(task_id=task_id)
    patient_str = format_patient(obs)
    trials_str = format_trials(obs)

    # Run the agent (re-reset internally)
    result = agent.run_episode(env, task_id=task_id)

    steps_str = format_steps(result.get("reasoning", []), result)
    result_str = format_result(result)

    return patient_str, trials_str, steps_str, result_str


# -----------------------------------------------
# GRADIO UI
# -----------------------------------------------

with gr.Blocks(title="ClinicalTrialMatchEnv") as demo:
    gr.Markdown(
        "# 🏥 ClinicalTrialMatchEnv\n"
        "### Watch an AI agent match a cancer patient to the right clinical trial\n"
        "Select a task difficulty and click **Run Episode** to see the agent reason step-by-step."
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=[f"{tid} — {desc}" for tid, desc in TASKS],
            value=f"{TASKS[0][0]} — {TASKS[0][1]}",
            label="Task",
        )
        run_btn = gr.Button("▶  Run Episode", variant="primary", scale=0)

    with gr.Row():
        with gr.Column():
            patient_box = gr.Textbox(label="Patient", lines=14, interactive=False)
            trials_box = gr.Textbox(label="Available Trials", lines=14, interactive=False)
        with gr.Column():
            steps_box = gr.Textbox(label="Agent Steps", lines=14, interactive=False)
            result_box = gr.Textbox(label="Final Result", lines=6, interactive=False)

    run_btn.click(
        fn=run_demo,
        inputs=[task_dropdown],
        outputs=[patient_box, trials_box, steps_box, result_box],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
