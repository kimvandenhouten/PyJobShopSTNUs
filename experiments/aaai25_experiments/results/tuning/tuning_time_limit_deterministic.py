# Import
import numpy as np
import pandas as pd
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark

from general.logger import get_logger
logger = get_logger(__name__)

# GENERAL SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'
INSTANCE_FOLDERS = ["ubo50", "ubo100"]
INSTANCE_IDS = range(1, 11)
perfect_information = True

data = []
for time_limit in [60, 600, 6000]:

    # Start solving the instances with perfect information
    for instance_folder in INSTANCE_FOLDERS:

        for instance_id in INSTANCE_IDS:
            np.random.seed(SEED)
            rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id)
            sample = rcpsp_max.durations
            logger.info(f'Start {instance_folder}_PSP{instance_id} timelimit {time_limit} and sample {sample}')
            res, _ = rcpsp_max.solve(sample, time_limit=time_limit, mode="Quiet")
            if res:
                logger.info(f'Res objective gap is {res.get_objective_gaps()[0]}')
                logger.info(f'Res objective value is {res.get_objective_values()[0]}')
                logger.info(f'Res solve time is {res.get_solve_time()}')
                logger.info(f'Res solver status {res.get_solve_status()}')

                data.append({"instance_folder": instance_folder, "instance_id": instance_id,
                             "sample_durations": sample, "time_limit": time_limit,
                             "obj_gap": res.get_objective_gaps()[0], "obj_value": res.get_objective_values()[0],
                             "solve_time": res.get_solve_time(), "solver_status": res.get_solve_status()})
            else:
                logger.info(f'Res solve time is {res.get_solve_time()}')
                logger.info(f'Res solver status {res.get_solve_status()}')

                data.append({"instance_folder": instance_folder, "instance_id": instance_id,
                             "sample_durations": sample, "time_limit": time_limit,
                                 "solve_time": res.get_solve_time(), "solver_status": res.get_solve_status()})

            data_df = pd.DataFrame(data)
            data_df.to_csv(f"experiments/aaai25_experiments/results/tuning/results_deterministic.csv")

