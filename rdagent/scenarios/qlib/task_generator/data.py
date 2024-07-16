import pickle
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.log import RDAgentLog
from rdagent.core.task_generator import TaskGenerator
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.utils.env import QTDockerEnv

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()
logger = RDAgentLog()

# class QlibFactorExpWorkspace:

#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf.yaml")

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def generate(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker for backtest results.
        """
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.generate(exp.based_experiments[-1])

        if RUNNER_SETTINGS.runner_cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp

        if exp.based_experiments:
            SOTA_factor = None
            if exp.based_experiments.__len__() != 1:
                SOTA_factor = self.process_factor_data(exp.based_experiments)

            # Process the new factors data
            new_factors = self.process_factor_data(exp)

            # Combine the SOTA factor and new factors if SOTA factor exists
            if SOTA_factor is not None and not SOTA_factor.empty:
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            # Sort and nest the combined factors under 'feature'
            combined_factors = combined_factors.sort_index()
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns

            # Save the combined factors to a pickle file
            combined_factors_path = DIRNAME / "env_factor/combined_factors_df.pkl"
            with open(combined_factors_path, "wb") as f:
                pickle.dump(combined_factors, f)

        #  Docker run
        # Call Docker, pass the combined factors to Docker, and generate backtest results
        qtde = QTDockerEnv()
        qtde.prepare()

        # Run the Docker command
        execute_log = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="rm -r mlruns")
        # Run the Qlib backtest
        execute_log = qtde.run(
            local_path=str(DIRNAME / "env_factor"),
            entry=f"qrun conf.yaml" if len(exp.based_experiments) == 0 else "qrun conf_combined.yaml",
        )

        execute_log = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="python read_exp_res.py")

        csv_path = DIRNAME / "env_factor/qlib_res.csv"

        if not csv_path.exists():
            logger.error(f"File {csv_path} does not exist.")
            return None

        result = pd.read_csv(csv_path, index_col=0).iloc[:, 0]

        exp.result = result
        if RUNNER_SETTINGS.runner_cache_result:
            self.dump_cache_result(exp, result)

        return exp

    def process_factor_data(self, exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
        """
        Process and combine factor data from experiment implementations.

        Args:
            exp (ASpecificExp): The experiment containing factor data.

        Returns:
            pd.DataFrame: Combined factor data without NaN values.
        """
        if isinstance(exp_or_list, QlibFactorExperiment):
            exp_or_list = [exp_or_list]
        factor_dfs = []

        # Collect all exp's dataframes
        for exp in exp_or_list:
            # Iterate over sub-implementations and execute them to get each factor data
            for implementation in exp.sub_implementations:
                message, df = implementation.execute(data_type="All")

                # Check if factor generation was successful
                if df is not None:
                    time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                    if pd.Timedelta(minutes=1) not in time_diff:
                        factor_dfs.append(df)

        # Combine all successful factor data
        if factor_dfs:
            return pd.concat(factor_dfs, axis=1)
        else:
            logger.error("No valid factor data found to merge.")
            return pd.DataFrame()  # Return an empty DataFrame if no valid data
