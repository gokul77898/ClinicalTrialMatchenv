---
title: ClinicalTrialMatchEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - clinical-trials
  - agent-environment
license: mit
---

# 🏥 ClinicalTrialMatchEnv

**An OpenEnv environment for AI agents to match cancer patients to clinical trials**

> Real-world task that oncology nurses spend hours on manually. Same constraints, same stakes.

## 🎯 What Is This?

An AI agent receives a patient chart (diagnosis, biomarkers, lab values, treatments) and must select the correct clinical trial from a pool of candidates. Wrong matches are penalized heavily (patient safety). The agent must investigate patient data, check trial eligibility criteria, and make the right decision within 20 steps.

**Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/OmilosAISolutions/ClinicalTrialMatchEnv)

## ⚡ Quick Start

### Run Locally (Docker)
```bash
docker build -t clinical-trial-env .
docker run -p 7860:7860 clinical-trial-env
# Visit http://localhost:7860/health
```

### Run Locally (Python)
```bash
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

### Test the Environment
```bash
# Run all 118 unit + integration tests
PYTHONPATH=/path/to/ClinicalTrialMatchEnv python -m pytest tests/

# Run baseline inference with LLM agent
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct:novita"
export HF_TOKEN="your_hf_token_here"
python inference.py
```

## 🎮 How It Works

1. **Agent calls `/reset`** → Gets patient chart + list of trials
2. **Agent investigates** → `{"type": "check_criteria", "trial_id": "TRIAL-LUNG-1234"}`
3. **Environment responds** → Inclusion/exclusion details, biomarker matches, eligibility summary
4. **Agent selects trial** → `{"type": "select_trial", "trial_id": "TRIAL-LUNG-1234"}`
5. **Agent resolves** → `{"type": "resolve"}` → Gets final grade (0.0-1.0)

**Key:** Correct trial = +1.0, Wrong trial = -1.0 (safety penalty), Efficiency bonus if ≤5 steps

## 📊 6 Tasks (Easy → Expert)

| Task | Difficulty | Trials | Baseline | Description |
|------|-----------|--------|----------|-------------|
| **single_match** | Easy | 3 | 0.90 | 1 eligible trial, obvious wrong candidates |
| **hidden_exclusion** | Medium | 5 | 0.73 | 2 trials pass inclusion but fail exclusion (traps) |
| **ambiguous_match** | Hard | 7 | 0.54 | 3 exclusion traps + 3 biomarker failures |
| **competing_trials** | Expert | 5 | 0.65 | 2 eligible trials, pick best based on trial_score |
| **contradictory_info** | Expert | 5 | 0.55 | Contradictory patient data, must flag before selecting |
| **multi_patient** | Expert | 5 | 0.00 | Match 3 patients simultaneously (hardest) |

**Baseline Model:** `meta-llama/Meta-Llama-3-70B-Instruct:novita`

## 🔧 API Reference

### POST /reset
Start new episode for a task
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "single_match"}'
```

### POST /step
Take an action
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "check_criteria", "trial_id": "TRIAL-LUNG-1234"}'
```

### GET /tasks
List all available tasks
```bash
curl http://localhost:7860/tasks
```

## 🎯 Actions Available

| Action | JSON Format | What It Does |
|--------|-------------|--------------|
| **check_criteria** | `{"type": "check_criteria", "trial_id": "TRIAL-XXX"}` | Check if patient is eligible for trial |
| **select_trial** | `{"type": "select_trial", "trial_id": "TRIAL-XXX"}` | Select this trial as the match |
| **flag_contradiction** | `{"type": "flag_contradiction", "reason": "..."}` | Flag contradictory patient data |
| **switch_case** | `{"type": "switch_case", "case_id": "case_2"}` | Switch to different patient (multi-patient mode) |
| **resolve** | `{"type": "resolve"}` | End episode and get final grade |

## 🏆 Reward Structure

| Event | Reward | Why |
|-------|--------|-----|
| ✅ Correct trial + efficient (≤5 steps) | +1.2 | Perfect performance |
| ✅ Correct trial selected | +1.0 | Safe match |
| ❌ Wrong trial selected | -1.0 | **Patient safety penalty** |
| 🔍 First check_criteria per trial | +0.05 | Encourages investigation |
| 🔁 Repeated action | -0.05 | Penalizes inefficiency |
| ⏱️ Max steps without resolve | -0.5 | Forces decision |

## 💡 Key Features

### Rich Eligibility Feedback
When you call `check_criteria`, you get:
- ✅ **Inclusion details**: Which rules passed/failed
- ❌ **Exclusion details**: Which exclusion rules triggered
- 🧬 **Biomarker details**: EGFR/ALK expression levels (0.0-1.0)
- 🏥 **Comorbidity details**: Severity-aware conflicts (mild/moderate/severe)
- 💊 **Prior treatment details**: Required/forbidden treatment matches
- 📝 **Plain English summary**: Human-readable eligibility explanation

### Advanced Trial Metadata
- **Capacity tracking**: `max_patients`, `enrolled_patients`, `has_capacity`
- **Deadline urgency**: `days_until_deadline`, `is_urgent` (≤14 days)
- **Quality scores**: `trial_score` (0.0-1.0) for comparing eligible trials

### Multi-Patient Mode
- Switch between 3 patients using `switch_case` action
- Each patient needs a different trial
- Tests agent's ability to manage multiple contexts

## 🧪 Testing & Validation

### Run All Tests (118 tests)
```bash
PYTHONPATH=/path/to/ClinicalTrialMatchEnv python -m pytest tests/ -v
```

**Test Coverage:**
- ✅ 40+ unit tests (eligibility engine, graders, task generation)
- ✅ 30+ integration tests (API endpoints, full episodes)
- ✅ End-to-end tests for all 6 tasks
- ✅ OpenEnv compliance validation

### Run Mock Inference (No API Key Needed)
```bash
PYTHONPATH=/path/to/ClinicalTrialMatchEnv python test_inference_mock.py
```
Uses hardcoded "perfect agent" responses to verify environment works correctly.

### Run Real Inference with LLM
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct:novita"
export HF_TOKEN="your_hf_token_here"
python inference.py
```

## 📁 Project Structure
```
ClinicalTrialMatchEnv/
├── api/
│   └── server.py              # FastAPI server (7860)
├── src/
│   ├── environment.py         # Main ClinicalTrialEnv class
│   ├── tasks.py               # 6 task definitions + generators
│   ├── graders.py             # Task-specific grading functions
│   ├── models.py              # Action, Observation, Reward models
│   ├── schemas/
│   │   ├── patient_schema.py  # Patient + biomarkers + lab trends
│   │   └── trial_schema.py    # ClinicalTrial + eligibility rules
│   └── engine/
│       └── eligibility_engine.py  # Rule-based eligibility checker
├── tests/
│   ├── unit/                  # 40+ unit tests
│   └── integration/           # 30+ integration tests
├── inference.py               # LLM baseline inference script
├── test_inference_mock.py     # Mock agent test (no API key)
├── openenv.yaml               # OpenEnv specification
├── Dockerfile                 # Production Docker image
└── requirements.txt           # Python dependencies
```

## 🚀 Deployment

**HuggingFace Space:** [https://huggingface.co/spaces/OmilosAISolutions/ClinicalTrialMatchEnv](https://huggingface.co/spaces/OmilosAISolutions/ClinicalTrialMatchEnv)

The environment is live and ready to use. No setup required — just call the API endpoints.

## 📊 Baseline Results

**Model:** `meta-llama/Meta-Llama-3-70B-Instruct:novita`

| Task | Grade | Correct | Steps | Notes |
|------|-------|---------|-------|-------|
| single_match | 0.90 | ✅ | 5 | Easy task, high success |
| hidden_exclusion | 0.73 | ✅ | 4 | Medium difficulty |
| ambiguous_match | 0.54 | ✅ | 17 | Hard, many traps |
| competing_trials | 0.65 | ✅ | 3 | Expert, quality reasoning |
| contradictory_info | 0.55 | ✅ | 4 | Expert, anomaly detection |
| multi_patient | 0.00 | ❌ | 17 | Expert, multi-context (hardest) |
| **Average** | **0.56** | **5/6** | | |

## 🤝 Contributing

This is an OpenEnv submission. For issues or improvements, please open a GitHub issue or PR.

## 📄 License

MIT License - See LICENSE file for details.
