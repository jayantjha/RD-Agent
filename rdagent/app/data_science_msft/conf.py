from typing import Literal

from pydantic_settings import SettingsConfigDict

from rdagent.app.data_science.conf import DataScienceBasePropSetting



class DataScienceBaseV2PropSetting(DataScienceBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="DS_", protected_namespaces=())

    # Main components
    ## Scen
    scen: str = "rdagent.scenarios.data_science.scen.KaggleScen"
    """Scenario class for data mining model"""

    ## Workflow Related
    consecutive_errors: int = 5

    debug_timeout: int = 600
    """The timeout limit for running on debugging data"""
    full_timeout: int = 3600
    """The timeout limit for running on full data"""

    ### specific feature

    #### enable specification
    spec_enabled: bool = True

    ### proposal related
    proposal_version: str = "v1"
    coder_on_whole_pipeline: bool = False
    max_trace_hist: int = 3

    coder_max_loop: int = 10
    runner_max_loop: int = 3

    session_root_path: str = ""
    """The root path of the session. It is used to load the session from the disk."""
    session_path: str = ""
    session_id: str = ""

    rule_base_eval: bool = False

    ### model dump
    enable_model_dump: bool = False
    enable_doc_dev: bool = False
    model_dump_check_level: Literal["medium", "high"] = "medium"

    azure_storage_account_url:str =""
    azure_storage_container_name:str=""

    log_experiment: bool = False
    log_manifest: bool = True
    upload_logs_to_storage: bool = True

DS_RD_SETTING = DataScienceBaseV2PropSetting()
