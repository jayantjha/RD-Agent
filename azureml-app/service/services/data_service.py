import os
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from typing import Union
from config import *
from logger import logger

def get_manifest_data(session_id: str, loop: int) -> Union[bytes, None]:
    
    # Construct the blob path
    blob_path = f"{session_id}/log/Loop_{loop}/manifest.json"
    return _get_blob_content(blob_path)

def get_file_content_data(session_id: str, path: str) -> Union[bytes, None]:
    # Construct the blob path
    blob_path = f"{session_id}/{path}"
    return _get_blob_content(blob_path)

def _get_blob_content(blob_path: str) -> bytes:

    logger.info(f"getting blob : {blob_path}")
    # Define the blob storage container name
    container_name = AZURE_STORAGE_CONTAINER_NAME

    # Initialize the BlobServiceClient using DefaultAzureCredential for managed identity
    
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=AZURE_STORAGE_ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    # Download the blob
    try:
        blob_client = container_client.get_blob_client(f"sessions/{blob_path}")
        download_stream = blob_client.download_blob()
        return download_stream.readall()
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve file content: {e}")