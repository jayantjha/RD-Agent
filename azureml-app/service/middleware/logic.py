from starlette.requests import Request
from starlette.responses import Response
from logger import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response: Response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

def add_cors_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change to specific origins in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
