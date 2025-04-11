import enum
import time
import uuid
import json

from rdagent.log import rdagent_logger as logger
from rdagent.core.experiment import RD_AGENT_SETTINGS

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Singleton for the project client
_project_client = None

def _get_project_client():
    """
    Create or return the existing AIProjectClient singleton instance.
    
    Returns:
        AIProjectClient: The initialized client
        
    Raises:
        ValueError: If connection string is not available
    """
    global _project_client
    
    if _project_client is None:
        if not RD_AGENT_SETTINGS.project_conn_string:
            raise ValueError("Project connection string is not set.")
            
        credential = DefaultAzureCredential()
        _project_client = AIProjectClient.from_connection_string(
            credential=credential,
            conn_str=RD_AGENT_SETTINGS.project_conn_string,
        )
    
    return _project_client

class TaskStatus(enum.Enum):
    STARTED = "STARTED"
    INPROGRESS = "IN_PROGRESS"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"

def publish_trace(task: str, status: TaskStatus, message_content: str, **kwargs) -> None:
    """
    Sends a message to the thread associated with this logger.

    :param message_content: The content of the message to send.
    """
    try:
        if not RD_AGENT_SETTINGS.thread_id:
            raise ValueError("Thread ID is not set.")
        
        payload_dict = {
            "task": task, 
            "status": status.value, 
            "message": message_content
        }
        if kwargs:
            payload_dict.update(kwargs)
        
        payload = json.dumps(payload_dict)
        
        project_client = _get_project_client()
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

def get_manual_approval(message_content: str) -> bool:
    """
    Get manual approval from the user for the given message.
    """
    if not RD_AGENT_SETTINGS.thread_id:
        return True

    try:
        request_id = str(uuid.uuid4()) 
        payload = json.dumps({"type": "approval", "requestId": request_id, "message": message_content})
        
        with _get_project_client() as project_client:
            message = project_client.agents.create_message(
                thread_id=RD_AGENT_SETTINGS.thread_id, 
                role="assistant", 
                content=payload)
            
            print("waiting for approval...", end='')

            while (1): 
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
        return True