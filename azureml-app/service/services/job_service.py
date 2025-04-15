from azure.identity import ManagedIdentityCredential, DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml.exceptions import ValidationException
from config import *
from logger import logger

class JobParameters:
    def __init__(self, agent_id: str, thread_id: str, user_prompt: str, data_uri: str, project_conn_string: str, competition_id: str):
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.user_prompt = user_prompt
        self.data_uri = data_uri
        self.project_conn_string = project_conn_string
        self.competition_id = competition_id

def submit_aml_job(job_params: JobParameters):
    try:
        logger.info(f"Preparing to submit AML job for agent_id: {job_params.agent_id} and thread_id: {job_params.thread_id}, competition_id: {job_params.competition_id}")

        # credential = ManagedIdentityCredential()
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=AML_WORKSPACE,
        )
        logger.info("Authenticated with Azure ML Client")
         # Merge ENV_VARS with additional environment variables
        environment_variables = {
            **ENV_VARS,  # Include all existing environment variables
            "THREAD_ID": job_params.thread_id,  # Add thread_id
            "PROJECT_CONN_STRING": job_params.project_conn_string  # Add project_conn_string
        }

        replaced_command = COMMAND_EXECUTE.replace("playground-series-s4e9", job_params.competition_id)

        command_job = command(
            display_name=f"agent-job-{job_params.thread_id}",
            description=f"Job for agent_id: {job_params.agent_id} and thread_id: {job_params.thread_id}",
            command=replaced_command,
            environment=ENVIRONMENT,
            compute=COMPUTE,
            experiment_name=EXPERIMENT_NAME,
            #inputs={"agent_id": job_params.agent_id},
            environment_variables=environment_variables,
        )

        ml_client.jobs.validate(command_job)
        job = ml_client.jobs.create_or_update(command_job)
        logger.info(f"Job submitted successfully: {job.name}")

        return {
            "job_id": job.name,
            "job_url": f"https://ml.azure.com/runs/{job.name}?wsid=/subscriptions/{SUBSCRIPTION_ID}/resourcegroups/{RESOURCE_GROUP}/workspaces/{AML_WORKSPACE}"
        }

    except ValidationException as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise ve
    except Exception as e:
        logger.error(f"Unexpected error during job submission: {str(e)}")
        raise e
