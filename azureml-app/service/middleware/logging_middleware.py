from starlette.requests import Request
from starlette.responses import Response
from logger import logger

async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response: Response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response
