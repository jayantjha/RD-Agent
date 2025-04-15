from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from datamodels.agent_request import AgentRequest
from logger import logger
from azure.ai.projects.aio import AIProjectClient
from services.job_service import JobParameters, submit_aml_job
from azure.identity.aio import DefaultAzureCredential
from config import FAKE_JOB, FAKE_THREAD_ID, PROJECT_CONNECTION_STRING, MODEL_DEPLOYMENT_NAME, AGENT_ID

router = APIRouter()

@router.post("/agents")
async def execute_agent(request: AgentRequest):
    try:
        logger.info(f"Received agent execution request: user_prompt='{request.user_prompt}', data_uri='{request.data_uri}'")

        if not request.user_prompt.strip():
            raise HTTPException(status_code=400, detail="user_prompt must not be empty")
        if not request.data_uri.strip():
            raise HTTPException(status_code=400, detail="data_uri must not be empty")

        logger.info("Agent execution logic would go here")

        agentid = AGENT_ID

        async with DefaultAzureCredential() as creds:
            async with AIProjectClient.from_connection_string(
                credential=creds, conn_str=PROJECT_CONNECTION_STRING
            ) as project_client:

                # agent = await project_client.agents.create_agent(
                #     model=MODEL_DEPLOYMENT_NAME,
                #     name="my-assistant",
                #     instructions="You are helpful assistant"
                # )
                # logger.info(f"Created agent, agent ID: {agent.id}")

                thread = await project_client.agents.create_thread()
                
                logger.info(f"Created thread, thread ID: {thread.id}")

                # Instantiate the JobParameters class
                job_params = JobParameters(
                    agent_id=agentid,
                    thread_id=thread.id,
                    user_prompt=request.user_prompt,
                    data_uri=request.data_uri,
                    project_conn_string=PROJECT_CONNECTION_STRING
                )
                submit_aml_job(job_params)
                thread_id = thread.id

                # TO BE HANDLED LATER

                message = await project_client.agents.create_message(
                    thread_id=thread.id,
                    role="assistant",
                    content="Initiating agent...",
                )
                logger.info(f"Created message, message ID: {message.id}")

                # run = await project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
                # logger.info(f"Run finished with status: {run.status}")

                # if run.status == "failed":
                #     logger.info(f"Run failed: {run.last_error}")

                # messages = await project_client.agents.list_messages(thread_id=thread.id)
                # logger.info(f"Messages: {messages}")

                # last_msg = messages.get_last_text_message_by_role(MessageRole.AGENT)
                # if last_msg:
                #     logger.info(f"Last Message: {last_msg.text.value}")

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "Agent task submitted",
                "agent_id": agentid,
                "thread_id": thread_id,
            }
)

    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
