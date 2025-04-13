import asyncio
import uuid
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config import PROJECT_CONNECTION_STRING
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from typing import Set, AsyncGenerator

router = APIRouter()

@router.get("/updates/{thread_id}")
async def get_updates(thread_id: str):
    async with DefaultAzureCredential() as creds:
        project_client = AIProjectClient.from_connection_string(
            credential=creds, conn_str=PROJECT_CONNECTION_STRING
        )
        
    return StreamingResponse(event_stream_polling(project_client, thread_id), media_type="text/event-stream")

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
            seen_content = set()  # Track unique content values
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
                            text_value = msg.content[0].text.value
                            # Skip if we've already seen this exact content
                            if text_value in seen_content:
                                continue
                            seen_content.add(text_value)
                            
                            parsed = json.loads(text_value)
                            parsed["createdAt"] = timestamp
                            parsed["id"] = msg.id
                            simplified_messages.append(parsed)
                        except json.JSONDecodeError as e:
                            print(f"Skipping message due to JSON error: {e}")
            
    return StreamingResponse(event_saved_stream(simplified_messages), media_type="text/event-stream")

async def event_stream_polling(project_client, thread_id: str):
    seen_ids: Set[str] = set()

    while True:
        try:
            new_messages = []

            async for page in get_all_messages_paged(project_client, thread_id, page_size=100):
                for msg in page.data:
                    if msg.id in seen_ids:
                        continue
                    seen_ids.add(msg.id)

                    try:
                        text_value = (
                            msg.content[0].text.value
                            if msg.content and hasattr(msg.content[0], 'text') and msg.content[0].text
                            else None
                        )
                        if text_value:
                            parsed = json.loads(text_value)
                            if hasattr(msg.created_at, 'timestamp'):
                                parsed["createdAt"] = int(msg.created_at.timestamp())
                                parsed["id"] = msg.id
                            new_messages.append(parsed)
                    except (AttributeError, IndexError, json.JSONDecodeError) as e:
                        print(f"Skipping message due to error: {e}")

            for msg in reversed(new_messages):
                await asyncio.sleep(1)
                yield f"data: {json.dumps(msg)}\n\n"

        except Exception as e:
            print(f"Polling error: {e}")

        await asyncio.sleep(30)


async def event_saved_stream(messages: list):
    for msg in messages:
        await asyncio.sleep(1)
        yield f"data: {json.dumps(msg)}\n\n"

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
