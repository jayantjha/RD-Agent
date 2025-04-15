from pydantic import BaseModel

class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str | None = None

