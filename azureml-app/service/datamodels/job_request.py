from pydantic import BaseModel

class JobRequest(BaseModel):
    agent_id: str
    thread_id: str
