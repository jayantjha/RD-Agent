import asyncio
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

async def event_stream():
    while True:
        await asyncio.sleep(2)
        guid = str(uuid.uuid4())
        message = f"data: {{\"id\": \"{guid}\", \"message\": \"This is a stream update.\"}}\n\n"
        yield message

@router.get("/updates")
async def get_updates():
    return StreamingResponse(event_stream(), media_type="text/event-stream")
