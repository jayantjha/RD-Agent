import os
from fastapi import APIRouter, HTTPException
from logger import logger
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.responses import StreamingResponse
from services.data_service import get_manifest_data, get_file_content_data

router = APIRouter()

@router.get("/manifest/{session_id}")
async def get_manifest(session_id: str, loop: int):
    logger.info(f"Received manifest request for session : {session_id}', loop='{loop}'")
    print(f"Received manifest request for session : {session_id}', loop='{loop}'")
    try:
        manifest = get_manifest_data(session_id, loop)
        if manifest is None:
            print("Manifest not found")
            raise HTTPException(status_code=404, detail="Manifest File not found")
        return JSONResponse(content=manifest.decode('utf-8'), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching manifest: {str(e)}")

@router.get("/file/{session_id}")
async def get_file_content(session_id: str, path: str):
    logger.info(f"Received file request for session : {session_id}', path='{path}'")
    print(f"Received manifest request for session : {session_id}', path='{path}'")
    try:
        content = get_file_content_data(session_id, path)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        return PlainTextResponse(content.decode('utf-8'), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching file content: {str(e)}")