from fastapi import FastAPI
#from fastapi.openapi.utils import get_openapi
from middleware.logging_middleware import log_requests
#from logger import logger
from controllers.job_controller import router as job_router
from controllers.agent_controller import router as agent_router
from controllers.data_controller import router as data_router

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello from Azure App Service - Updated! ✅"}

# Custom OpenAPI schema version - add below only if open API schema is to be generated
# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Job API",
#         version="1.0.0",
#         description="Execute an Azure Machine Learning job",
#         routes=app.routes,
#     )
#     openapi_schema["openapi"] = "3.0.3"
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi

app.middleware("http")(log_requests)
app.include_router(job_router, prefix="/jobs", tags=["Jobs"])
app.include_router(agent_router, prefix="/execute", tags=["Agents"])
app.include_router(data_router, prefix="/data", tags=["Data"])
