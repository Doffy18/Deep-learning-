from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import start_email_agent, resume_email_agent

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)


class StartRequest(BaseModel):
    thread_id: str
    prompt: str

class ResumeRequest(BaseModel):
    thread_id: str
    feedback: str

@app.post("/agent/start")
def handle_start(payload: StartRequest):
    # Triggers graph generation, hits interrupt, returns draft JSON instantly
    response = start_email_agent(payload.thread_id, payload.prompt)
    return response

@app.post("/agent/resume")
def handle_resume(payload: ResumeRequest):
    # Injects user feedback, decides to finish or rewrite, returns next state
    response = resume_email_agent(payload.thread_id, payload.feedback)
    return response