import asyncio
import uuid
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config import PROJECT_CONNECTION_STRING
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential

router = APIRouter()

async def event_stream():
    while True:
        await asyncio.sleep(2)
        guid = str(uuid.uuid4())
        
        message = f"data: {{\"task\": \"CODING\", \"status\": \"STARTED\", \"createdAt\": 1744318179,  \"message\": \"Starting code implementation\"}}\n\n"
        yield message

async def event_saved_stream(messages: list):
    for msg in messages:
        await asyncio.sleep(1)
        yield f"data: {msg}\n\n"

@router.get("/updates/{thread_id}")
async def get_updates(thread_id: str):
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/updates/saved/{thread_id}")
async def get_saved_updates(thread_id: str):
    async with DefaultAzureCredential() as creds:

        async with AIProjectClient.from_connection_string(
            credential=creds, conn_str=PROJECT_CONNECTION_STRING
        ) as project_client:

            all_messages = []
            
            # Collect all messages page by page
            async for page in get_all_messages_paged(project_client, thread_id, page_size=100):
                all_messages.extend(page.data)
            all_messages.reverse()
            print(f"Total messages retrieved: {len(all_messages)}")
            
            # Create new simplified JSON with only the content.text.value
            simplified_messages = []
            for msg in all_messages:
                # Check if message has content and text value
                if (msg.content and 
                    len(msg.content) > 0 and 
                    hasattr(msg.content[0], 'text') and 
                    msg.content[0].text and 
                    hasattr(msg.content[0].text, 'value')):
                    # Convert timestamp to a string or integer if it's a datetime object
                    timestamp = msg.created_at
                    if hasattr(timestamp, 'timestamp'):  # Check if it's a datetime object
                        timestamp = int(timestamp.timestamp())  # Convert to Unix timestamp (integer)
                    # Extract timestamp and value
                    if msg.content[0].text.value:
                        try:
                            parsed = json.loads(msg.content[0].text.value)
                            parsed["createdAt"] = timestamp
                            simplified_messages.append(parsed)
                        except json.JSONDecodeError as e:
                            print(f"Skipping message due to JSON error: {e}")
            
    return StreamingResponse(event_saved_stream(simplified_messages), media_type="text/event-stream")

async def get_all_messages_paged(project_client, thread_id, page_size=20):
    # Get the first page
    messages = await project_client.agents.list_messages(
        thread_id=thread_id, 
        limit=page_size,
        order="desc"  # Newest messages first, use "asc" for oldest first
    )
    
    print(f"Page 1 - Retrieved {len(messages.data)} messages")
    yield messages
    
    # Continue fetching pages until no more messages
    page_num = 2
    while messages.has_more:
        # Use the last ID from the current page as the "after" cursor for the next page
        last_message_id = messages.data[-1].id
        
        # Fetch the next page
        messages = await project_client.agents.list_messages(
            thread_id=thread_id,
            limit=page_size,
            order="desc",
            after=last_message_id
        )
        
        print(f"Page {page_num} - Retrieved {len(messages.data)} messages")
        yield messages
        page_num += 1
