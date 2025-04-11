from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from datamodels.job_request import JobRequest
# from services.job_service import submit_aml_job
from logger import logger

router = APIRouter()

@router.post("/execute")
async def execute_job(request: JobRequest):
    try:
        logger.info(f"Received job execution request for agent_id: {request.agent_id} and thread_id: {request.thread_id}")

        if not isinstance(request.agent_id, str):
            raise HTTPException(status_code=400, detail="agent_id must be a string")
        if not isinstance(request.thread_id, str):
            raise HTTPException(status_code=400, detail="thread_id must be a string")

        # TO BE IMPLEMENTED IF NEEDED
        # Instantiate the JobParameters class
        # job_params = JobParameters(
        #     agent_id=agent.id,
        #     thread_id=thread.id,
        #     user_prompt=request.user_prompt,
        #     data_uri=request.data_uri,
        #     project_conn_string=PROJECT_CONNECTION_STRING
        # )

        # result = submit_aml_job(job_params)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "Job submitted",
                # "job_id": result["job_id"],
                # "job_url": result["job_url"]
            }
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
