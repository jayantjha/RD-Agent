import asyncio
import uuid
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from azure.identity.aio import DefaultAzureCredential
from datamodels.chat_request import ChatRequest
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import MessageRole
from logger import logger
from config import PROJECT_CONNECTION_STRING, CHAT_START_AGENT_ID
from azure.ai.projects.models import RequiredFunctionToolCall, SubmitToolOutputsAction, ToolOutput
from utils.tool_functions import tool_functions

router = APIRouter()

@router.post("")
async def start_chat(request: ChatRequest):
    try:
        logger.info(f"Received start chat request for thread_id: {request.thread_id}, user_message: {request.message}")

        async with DefaultAzureCredential() as creds:
            async with AIProjectClient.from_connection_string(
                credential=creds, conn_str=PROJECT_CONNECTION_STRING
            ) as project_client:
                thread_id = request.thread_id
                if not request.thread_id or not isinstance(request.thread_id, str) or not request.thread_id.strip():
                    # Create a new thread for this chat session
                    thread = await project_client.agents.create_thread()
                    thread_id = thread.id
                    logger.info(f"Created thread for chat, thread ID: {thread.id}")
                else:
                    # Validate the existing thread ID
                    thread = await project_client.agents.get_thread(request.thread_id)
                    if not thread:
                        raise HTTPException(status_code=404, detail="Thread not found")
                    logger.info(f"Using existing thread, thread ID: {thread.id}")
                
                if request.message and isinstance(request.message, str):
                    # Add the user's message to the thread
                    message = await project_client.agents.create_message(
                        thread_id=thread_id,
                        role="user",
                        content=request.message,
                    )
                    logger.info(f"Created message, message ID: {message.id}")
                
                # Process the message with the agent
                run = await project_client.agents.create_run(
                    thread_id=thread.id, 
                    agent_id=CHAT_START_AGENT_ID
                )
                logger.info(f"Run finished for run: {run.id} with status: {run.status}")
                

                if run.status == "failed":
                    logger.error(f"Run failed: {run.last_error}")
                    raise HTTPException(status_code=500, detail=f"Agent run failed: {run.last_error}")
                
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "run_id": run.id,
                        "thread_id": thread.id,
                    }
                )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Start chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{thread_id}/{run_id}")
async def stream_chat(thread_id: str, run_id: str):
    try:
        logger.info(f"Received stream chat request for run_id: {run_id}, thread_id: {thread_id}")

        if not run_id or not isinstance(run_id, str):
            raise HTTPException(status_code=400, detail="run_id must not be empty")
        
        if not thread_id or not isinstance(thread_id, str):
            raise HTTPException(status_code=400, detail="thread_id must not be empty")
        
        async def generate_response():
            try:
                async with DefaultAzureCredential() as creds:
                    async with AIProjectClient.from_connection_string(
                        credential=creds, conn_str=PROJECT_CONNECTION_STRING
                    ) as project_client:
                        completed = False
                        while not completed:
                            run = await project_client.agents.get_run(
                                run_id=run_id,
                                thread_id=thread_id
                            )
                            if run.status == "failed":
                                logger.error(f"Run failed: {run.last_error}")
                                yield f"data: Error: {run.last_error}\n\n"
                                break
                            elif run.status == "completed":
                                # Mark as completed but don't break - we need to fetch messages first
                                completed = True
                            elif run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                                if not tool_calls:
                                    logger.info("No tool calls provided - cancelling run")
                                    await project_client.agents.cancel_run(thread_id=thread_id, run_id=run.id)
                                    break

                                tool_outputs = []
                                for tool_call in tool_calls:
                                    if isinstance(tool_call, RequiredFunctionToolCall):
                                        try:
                                            logger.info(f"Executing tool call: {tool_call}")
                                            output = tool_functions.execute(tool_call)
                                            tool_call.function.name
                                            tool_outputs.append(
                                                ToolOutput(
                                                    tool_call_id=tool_call.id,
                                                    output=output,
                                                )
                                            )
                                        except Exception as e:
                                            logger.info(f"Error executing tool_call {tool_call.id}: {e}")

                                logger.info(f"Tool outputs: {tool_outputs}")
                                if tool_outputs:
                                    await project_client.agents.submit_tool_outputs_to_run(
                                        thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                                    )
                                    await project_client.agents.update_thread(
                                        thread_id=thread_id,
                                        metadata={
                                            tool_call.function.name: output,
                                        }
                                    )

                            # If not completed, wait and continue the loop
                            if not completed:
                                await asyncio.sleep(1)
                                continue
                                
                            # If we're here, the run is completed and we can fetch messages
                            try:
                                messages = await project_client.agents.list_messages(thread_id=thread_id, limit=5)
                                message = messages.get_last_message_by_role(MessageRole.AGENT)
                                yield f"data: {message.content[0].text.value}\n\n"
                            except Exception as e:
                                logger.error(f"Error getting messages: {str(e)}")
                                yield f"data: Error getting agent response: {str(e)}\n\n"
                            break
                        
            except Exception as e:
                logger.error(f"Error in stream generation: {str(e)}")
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream"
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Stream chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
