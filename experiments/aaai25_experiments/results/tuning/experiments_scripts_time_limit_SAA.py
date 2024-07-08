# Import
import numpy as np
import pandas as pd
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
import ast

# Import reactive approach
from experiments.aaai25_experiments.run_reactive_approach import run_reactive_offline, run_reactive_online

# Import proactive approach
from experiments.aaai25_experiments.run_proactive_approach import run_proactive_offline, run_proactive_online

# Import STNU approach
from experiments.aaai25_experiments.run_stnu_approach import run_stnu_offline, run_stnu_online


from general.logger import get_logger
logger = get_logger(__name__)

# GENERAL SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'
INSTANCE_FOLDERS = ["ubo50", "ubo100"]
INSTANCE_IDS = range(1, 11)
nb_scenarios_test = 10
perfect_information = False
reactive = True
proactive = True
stnu = True
noise_factor = 1
writing = True

def check_pi_feasible(instance_folder, instance_id, sample_index, duration_sample, noise_factor):
    df = pd.read_csv(f'experiments/aaai25_experiments/results/results_pi_{instance_folder}_{noise_factor}.csv')
    filtered_df = df[(df['instance_id'] == instance_id) & (df['sample'] == sample_index)]
    assert len(filtered_df) == 1

    solve_result = filtered_df["solver_status"].tolist()[0]

    if solve_result in ["Infeasible", "Unknown"]:
        feasible = False
    else:
        feasible = True

    logger.debug(f'{instance_folder}_PSP{instance_id} with {duration_sample} PI feasibility is {feasible}')

    if np.random.randint(0, 100) < 100:
        real_duration = filtered_df["sample_durations"].tolist()[0]
        real_duration = ast.literal_eval(real_duration)
        logger.debug(f'pi duration {real_duration} and duration sample is {duration_sample}')
        assert real_duration == duration_sample

    if feasible:
        obj_pi = filtered_df['obj_value'].to_list()[0]
    else:
        obj_pi = np.inf
    return feasible, obj_pi

data = []
output_file = f'experiments/aaai25_experiments/results/results_time_limit_SAA={noise_factor}.csv'

if proactive:
    # RUN PROACTIVE EXPERIMENTS
    # Settings proactive approach

    for (mode, time_limit, nb_scenarios_saa) in [("SAA_smart", 60, 4), ("SAA_smart", 600, 4), ("SAA_smart", 3600, 4)]:
        # Run the experiments
        for instance_folder in INSTANCE_FOLDERS:
            for instance_id in INSTANCE_IDS:
                np.random.seed(SEED)
                rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor)
                test_durations_samples = rcpsp_max.sample_durations(nb_scenarios_test)
                data_dict = run_proactive_offline(rcpsp_max, time_limit, mode, nb_scenarios_saa)

                for i, duration_sample in enumerate(test_durations_samples):
                    pi_feasible, obj_pi = check_pi_feasible(instance_folder, instance_id, i, duration_sample, noise_factor)
                    data_dict["obj_pi"] = obj_pi
                    if pi_feasible:
                        data += run_proactive_online(rcpsp_max, duration_sample, data_dict)
                        data_df = pd.DataFrame(data)
                        if writing:
                            data_df.to_csv(output_file, index=False)
                    else:
                        logger.info(f'Instance {rcpsp_max.instance_folder}PSP{rcpsp_max.instance_id}, sample {i}: '
                                    f'We can skip the proactive approach')
