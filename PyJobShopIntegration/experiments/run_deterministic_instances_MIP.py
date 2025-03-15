# Import
import numpy as np
import pandas as pd
from rcpsp_max.solvers.RCPSP_MIP_benchmark import RCPSP_MIP_Benchmark
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
from general.logger import get_logger
logger = get_logger(__name__)

"""
This script contains the experiments for solving the deterministic instances with perfect information. Note that 
the setting "writing=False" means that no csv is written while running this script. Please be aware that the script
run_experiments.py uses the csv files with the perfect information results to check whether the perfect information
problem is feasible before running one of the stochastic scheduling methods.
"""

# GENERAL SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'
INSTANCE_FOLDERS = ["j10", "j20", "j30", "ubo50", "ubo100"]
INSTANCE_IDS = range(1, 51)
time_limit = 600
writing = True
data = []


for instance_folder in INSTANCE_FOLDERS:
    for instance_id in INSTANCE_IDS:
        print('\n')
        # SOLVE CP
        rcpsp_max_cp = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id,
                                                       noise_factor=1)
        logger.info(f'CP: Start {instance_folder}_PSP{instance_id} timelimit {time_limit}')
        res, _ = rcpsp_max_cp.solve(time_limit=time_limit)
        solve_time_cp = res.get_solve_time()
        terminal_cp = res.get_solve_status()
        logger.info(f'Solve time is {solve_time_cp}')
        logger.info(f'Solver status {terminal_cp}')
        if res:
            objective_cp = res.get_objective_values()[0]
            logger.info(f'Objective value is {objective_cp}')
        else:
            objective_cp = None

        # SOLVE MIP
        rcpsp_max_mip = RCPSP_MIP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor=1)
        logger.info(f'MIP: Start {instance_folder}_PSP{instance_id} timelimit {time_limit}')
        solve_time_mip, terminal_mip, objective_mip = rcpsp_max_mip.solve_MIP(time_limit=time_limit)

        # Store data
        data.append(
            {"instance_folder": instance_folder, "instance_id": instance_id, "time_limit": time_limit,
             "solve_time_cp": solve_time_cp, "solve_time_mip": solve_time_mip,
             "objective_cp": objective_cp, "objective_mip": objective_mip,
             "status_cp": terminal_cp, "status_mip": terminal_mip}
        )

        data_df = pd.DataFrame(data)
        if writing:
            data_df.to_csv(f"results_mip_versus_cp_{time_limit}.csv")




