from pydantic import BaseModel

class FileManifestItem(BaseModel):
    file_name: str
    path: str

class FileManifestResponse(BaseModel):
    files: list[FileManifestItem]