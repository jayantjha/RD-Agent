import enum
import time
import uuid
import json
from typing import Optional, Dict, Any

from rdagent.log import rdagent_logger as logger
from rdagent.core.experiment import RD_AGENT_SETTINGS

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

class TaskStatus(enum.Enum):
    STARTED = "STARTED"
    INPROGRESS = "IN_PROGRESS"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"

class FoundryAgent:
    """
    Singleton class for interacting with Azure AI Project threads.
    Maintains state and handles communication with the project client.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FoundryAgent, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._project_client = None
            self._loop_count = None
            self._session_id = None
            self._initialized = True
    
    @property
    def loop_count(self) -> int:
        """Get the current loop count."""
        return self._loop_count
    
    def set_loop_count(self, count: int) -> None:
        """Set the loop count to a specific value."""
        self._loop_count = count

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id
    
    def set_session_id(self, session_id: str) -> None:
        """Set the session ID to a specific value."""
        self._session_id = session_id
    
    def get_project_client(self):
        """
        Create or return the existing AIProjectClient instance.
        
        Returns:
            AIProjectClient: The initialized client
            
        Raises:
            ValueError: If connection string is not available
        """
        if self._project_client is None:
            if not RD_AGENT_SETTINGS.project_conn_string:
                raise ValueError("Project connection string is not set.")
                
            credential = DefaultAzureCredential()
            self._project_client = AIProjectClient.from_connection_string(
                credential=credential,
                conn_str=RD_AGENT_SETTINGS.project_conn_string,
            )
        
        return self._project_client
    
    def publish_trace(
            self,
            task: str, 
            status: TaskStatus, 
            message_content: str, 
            description: str = None, 
            **kwargs) -> None:
        """
        Sends a message to the thread associated with this logger.
        
        Args:
            task: The task name
            status: The task status
            message_content: The content of the message to send
            description: Optional description
            kwargs: Additional parameters to include in the payload
        """
        try:
            if not RD_AGENT_SETTINGS.thread_id:
                raise ValueError("Thread ID is not set.")
            
            payload_dict = {
                "task": task, 
                "status": status.value, 
                "message": message_content
            }

            if self.loop_count is not None:
                payload_dict["loop_count"] = self.loop_count
            if description:
                payload_dict["description"] = description
            if self.session_id:
                payload_dict["session_id"] = self.session_id
            if kwargs:
                payload_dict.update(kwargs)
            
            payload = json.dumps(payload_dict)
            
            project_client = self.get_project_client()
            message = project_client.agents.create_message(
                thread_id=RD_AGENT_SETTINGS.thread_id,
                role="assistant",
                content=payload,
            )

            if not message:
                raise ValueError(f"Failed to pass message to thread.")

        except Exception as e:
            # Log the exception object
            logger.info(e, tag="send_message_to_thread_error")

    def get_manual_approval(self, message_content: str) -> bool:
        """
        Get manual approval from the user for the given message.
        
        Args:
            message_content: The message to request approval for
            
        Returns:
            bool: True if approved, False otherwise
        """
        if not RD_AGENT_SETTINGS.thread_id:
            return True

        try:
            request_id = str(uuid.uuid4())
            payload = json.dumps({
                "type": "approval", 
                "requestId": request_id, 
                "message": message_content
            })
            
            with self.get_project_client() as project_client:
                message = project_client.agents.create_message(
                    thread_id=RD_AGENT_SETTINGS.thread_id, 
                    role="assistant", 
                    content=payload)
                
                print("waiting for approval...", end='')

                while True: 
                    print(".", end='')
                    time.sleep(10)
                    messages = project_client.agents.list_messages(thread_id=RD_AGENT_SETTINGS.thread_id)
                    last_message = messages.get_last_text_message_by_role(role="user") 
                    if last_message is None or last_message.text is None or last_message.text.value is None:
                        continue

                    try:
                        response = json.loads(last_message.text.value)
                        if response.get("type") == "approval" and response.get("requestId") == request_id: 
                            return response.get("approval", False)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.info(e, tag="get_manual_approval_error")
            return True


foundry = FoundryAgent()

def publish_trace(
        task: str, 
        status: TaskStatus, 
        message_content: str, 
        description: str = None, 
        **kwargs) -> None:
    foundry.publish_trace(task, status, message_content, description, **kwargs)

def get_manual_approval(message_content: str) -> bool:
    return foundry.get_manual_approval(message_content)

def _get_project_client():
    return foundry.get_project_client()