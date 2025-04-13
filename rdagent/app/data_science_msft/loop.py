import json
from pathlib import Path
from typing import Any, Optional

import fire
from pydantic import BaseModel

from rdagent.app.data_science_msft.conf import DS_RD_SETTING
from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import CoderError, RunnerError
from rdagent.core.proposal import ExperimentFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from rdagent.utils.foundry_agent import TaskStatus, publish_trace, foundry
from rdagent.scenarios.kaggle.kaggle_crawler import download_data
import uuid
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import jsonpickle

class File(BaseModel):
    description: Optional[str] = None
    path: Optional[str] = None
    name: Optional[str] = None
    size: Optional[int] = None

class Model(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None
    files: Optional[list[File]] = None

class Manifest(BaseModel):
    loop_id: int
    step_name: str
    model: Optional[Model] = None
    files: Optional[list[File]] = None
    summary: Optional[str] = None
    workspace_path: Optional[str] = None



class DataScienceV2RDLoop(DataScienceRDLoop):
    skip_loop_error = (CoderError, RunnerError)
    
    def __init__(self, *args, **kwargs):
        self.manifest = None
        super().__init__(*args, **kwargs)

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        exp: DSExperiment = super().direct_exp_gen(prev_out)
        self._log(exp)
        return exp

    def coding(self, prev_out: dict[str, Any]):
        exp: DSExperiment = super().coding(prev_out)
        self._log(exp)
        return exp
    
    def running(self, prev_out: dict[str, Any]):
        exp: DSExperiment = super().running(prev_out)
        self._log(exp)
        return exp

    def feedback(self, prev_out: dict[str, Any]) -> ExperimentFeedback:
        exp: DSExperiment = super().feedback(prev_out)
        self._log(exp)
        return exp

    def record(self, prev_out: dict[str, Any]):
        super().record(prev_out)

    def _log(self, exp: DSExperiment):
        step_name = self.steps[self.step_idx]
        self._write_experiment_data(exp, step_name)
     
        self._write_experiment_manifest(exp, step_name)

    def _write_experiment_manifest(self, exp, step_name):
        if not DS_RD_SETTING.log_manifest:
            return
        manifest = Manifest(loop_id=self.loop_idx, step_name=self.steps[self.step_idx])
        if hasattr(exp, 'experiment_workspace') and exp.experiment_workspace and exp.experiment_workspace.workspace_path.exists():
            manifest.workspace_path = str(exp.experiment_workspace.workspace_path.relative_to(DS_RD_SETTING.session_path))
            manifest.files = [File(name=f.name, path=str(f.relative_to(DS_RD_SETTING.session_path).parent), size=f.stat().st_size) for f in exp.experiment_workspace.workspace_path.iterdir() if f.is_file()]
            
            model_path = exp.experiment_workspace.workspace_path / "models"
            if model_path.exists():
                model_files = [File(name=f.name, path=str(f.relative_to(DS_RD_SETTING.session_path).parent), size=f.stat().st_size) for f in model_path.iterdir() if f.is_file()]
                manifest.model = Model(
                    name="model",
                    files=model_files,
                    size=sum(file.size for file in model_files)
                )
            self._log_object(manifest, "manifest", step_name)
            self._log_object(manifest, "manifest", step_name, Path(DS_RD_SETTING.session_path) / "log" / f"Loop_{self.loop_idx}")
        publish_trace("MANIFEST_CREATED", TaskStatus.COMPLETED, f"Manifest created for loop {self.loop_idx}, step {step_name}")


    def _write_experiment_data(self, exp, step_name):
        if not DS_RD_SETTING.log_experiment:
            return
        self._log_object(exp, "experiment", step_name)
        publish_trace("EXPERIMENT_LOGGED", TaskStatus.COMPLETED, f"Experiment logged for loop {self.loop_idx}, step {step_name}")
        

    def _log_object(self, obj:object, name:str, step_name:str, path:Optional[Path]=None):
        if path is None:
            path = Path(DS_RD_SETTING.session_path) / "log" / f"Loop_{self.loop_idx}"/ f"{step_name}" / f"{name}.json"
        else:
            path = path / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:   
            try:
                if isinstance(obj, BaseModel):
                    f.write(obj.model_dump_json())
                else:
                    f.write(jsonpickle.encode(obj, unpicklable=True))
            except TypeError:
                logger.error(f"Error in logging object: {name} with type {type(obj)}")
        

class FolderWatcher(FileSystemEventHandler):
    def __init__(self, folder_path, session_id):
        self.session_id = session_id
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=DS_RD_SETTING.azure_storage_account_url, credential=credential)
        self.container_client = blob_service_client.get_container_client(DS_RD_SETTING.azure_storage_container_name)
        self.folder_path = Path(folder_path)

        try:
            self.container_client.upload_blob(
                name=f"sessions/{self.session_id}/test.txt", 
                data="this is test", 
                overwrite=True,
                content_type="text/plain"
            )
            logger.info(f"Uploaded test")
        except Exception as e:
           return



    def on_modified(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if file_path.suffix not in {".json", ".py", ".csv", ".md"}:
            return
        
        # Determine content type based on file extension
        content_type_map = {
            ".json": "application/json",
            ".py": "text/x-python",
            ".csv": "text/csv",
            ".md": "text/markdown",
            ".txt": "text/plain",
        }
        content_type = content_type_map.get(file_path.suffix, "application/octet-stream")
        
        blob_name = file_path.relative_to(self.folder_path).as_posix()
        try:
            with open(file_path, "rb") as data:
                self.container_client.upload_blob(
                    name=f"sessions/{self.session_id}/{blob_name}", 
                    data=data, 
                    overwrite=True,
                    content_type=content_type
                )
                logger.info(f"Uploaded {file_path} to blob storage as {blob_name} with content type {content_type}")
                if file_path.name == "main.py":
                    publish_trace("DS_UPLOADED", TaskStatus.COMPLETED, f"Uploaded {file_path} to blob storage as {blob_name}", self.session_id)
        except Exception as e:
           return

def watch_daemon(session_id):
    if not DS_RD_SETTING.upload_logs_to_storage:
        return
    stop_event = threading.Event()
    thread = threading.Thread(target=watch, args=(session_id, stop_event), daemon=True)
    thread.start()
    return thread, stop_event

def watch(session_id, stop_event):
    folder_to_watch = Path(DS_RD_SETTING.session_root_path) / session_id
    if not folder_to_watch.exists():
        folder_to_watch.mkdir(parents=True, exist_ok=True)
    event_handler = FolderWatcher(folder_path=folder_to_watch, session_id=session_id)
    observer = Observer()
    observer.schedule(event_handler, path=str(folder_to_watch), recursive=True)
    observer.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        observer.stop()
        observer.join()

def main(
    path=None, output_path=None, step_n=None, loop_n=None, competition="bms-molecular-translation", do_truncate=True
):
    """

    Parameters
    ----------
    path :
        path like `$LOG_PATH/__session__/1/0_propose`. It indicates that we restore the state that after finish the step 0 in loop 1
    output_path :
        path like `$LOG_PATH`. It indicates that where we want to save our session and log information.
    step_n :
        How many steps to run; if None, it will run forever until error or KeyboardInterrupt
    loop_n :
        How many loops to run; if None, it will run forever until error or KeyboardInterrupt
        - if current loop is incomplete, it will be counted as the first loop for completion.
        - if both step_n and loop_n are provided, the process will stop as soon as either condition is met.
    competition :
    do_truncate :
        If set to True, the logger will truncate the future log messages by calling `logger.storage.truncate`.


    Auto R&D Evolving loop for models in a Kaggle scenario.
    You can continue running session by
    .. code-block:: bash
        dotenv run -- python rdagent/app/data_science/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional parameter
        rdagent kaggle --competition playground-series-s4e8  # You are encouraged to use this one.
    """
    if competition is not None:
        DS_RD_SETTING.competition = competition

    if DS_RD_SETTING.session_root_path:
        initialize_session()

    if DS_RD_SETTING.competition:
        if DS_RD_SETTING.scen.endswith("KaggleScen"):
            download_data(competition=DS_RD_SETTING.competition, settings=DS_RD_SETTING)
        else:
            if not Path(f"{DS_RD_SETTING.local_data_path}/{competition}").exists():
                logger.error(f"Please prepare data for competition {competition} first.")
                return
    else:
        logger.error("Please specify competition name.")
    if path is None:
        kaggle_loop = DataScienceV2RDLoop(DS_RD_SETTING)
    else:
        kaggle_loop = DataScienceV2RDLoop.load(path, output_path, do_truncate)
    kaggle_loop.run(step_n=step_n, loop_n=loop_n)

def initialize_session():
    DS_RD_SETTING.session_id = str(uuid.uuid4())
    foundry.set_session_id(DS_RD_SETTING.session_id)
    DS_RD_SETTING.session_path = f"{DS_RD_SETTING.session_root_path}/{DS_RD_SETTING.session_id}"
    logger.set_trace_path(Path(DS_RD_SETTING.session_path) / "log")
    RD_AGENT_SETTINGS.workspace_path = Path(DS_RD_SETTING.session_path) / "workspace"
    watch_daemon(DS_RD_SETTING.session_id)


if __name__ == "__main__":
    fire.Fire(main)


