from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.environment import ClinicalTrialEnv
from src.models import Action, Observation, Reward
from src.tasks import list_tasks

app = FastAPI(
    title="ClinicalTrialMatchEnv",
    description="OpenEnv environment: AI agent matches cancer patients to clinical trials",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = ClinicalTrialEnv(num_trials=5, max_steps=20)

class ResetRequest(BaseModel):
    patient_seed: Optional[int] = None
    trial_seed: Optional[int] = None
    task_id: Optional[str] = None

@app.get("/")
async def root():
    return {
        "name": "ClinicalTrialMatchEnv",
        "version": "1.0.0",
        "description": "OpenEnv environment for clinical trial matching",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "environment": "ClinicalTrialMatchEnv"}

@app.post("/reset")
async def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    try:
        obs = env.reset(
            patient_seed=request.patient_seed,
            trial_seed=request.trial_seed,
            task_id=request.task_id
        )
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
async def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/state")
async def state():
    try:
        obs = env.state()
        return obs.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks")
async def tasks():
    task_list = list_tasks()
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "num_trials": t.num_trials
            }
            for t in task_list
        ]
    }
