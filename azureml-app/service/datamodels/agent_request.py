from pydantic import BaseModel

class AgentRequest(BaseModel):
    user_prompt: str
    data_uri: str
