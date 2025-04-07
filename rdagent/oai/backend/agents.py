from typing import Any
from rdagent.oai.backend.deprec import DeprecBackend
from rdagent.oai.llm_conf import LLM_SETTINGS
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

class AgentsAPIBackend(DeprecBackend):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.thread_id = None
        self.coder_agent_id = LLM_SETTINGS.coder_agent_id
        self.research_agent_id = LLM_SETTINGS.research_agent_id
        self.project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(), conn_str=LLM_SETTINGS.project_connection_string
        )   
        super().__init__(*args, **kwargs)

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        thread_id = self.thread_id
        if thread_id is None:
            thread = self.project_client.agents.create_thread()
            thread_id = thread.id
            self.thread_id = thread_id
        
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]
        else:
            agent_name = "ResearchAgent"
        
        if agent_name == "CoderAgent":
            agent_id = self.coder_agent_id
        else:
            agent_id = self.research_agent_id

        for message in messages:
            if message["role"] == LLM_SETTINGS.system_prompt_role:
                message["role"] = "user"
            self.project_client.agents.create_message(
                thread_id=self.thread_id,
                content=message["content"],
                role=message["role"],
            )
        self.project_client.agents.create_and_process_run(thread_id=thread_id, agent_id=agent_id)
        last_msg = self._get_last_message(thread_id)
        return last_msg, None

    def _get_last_message(self, thread_id:str) -> str:
        messages = self.project_client.agents.list_messages(thread_id=thread_id, limit=10)
        last_msg = messages.get_last_text_message_by_role("assistant")
        return last_msg.text.value
    

