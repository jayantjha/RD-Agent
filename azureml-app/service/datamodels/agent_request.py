from pydantic import BaseModel

class AgentRequest(BaseModel):
    user_prompt: str | None
    data_uri: str | None
    chat_thread_id: str | None
