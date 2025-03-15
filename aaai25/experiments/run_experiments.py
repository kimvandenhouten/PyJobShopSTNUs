# Import
import numpy as np
import pandas as pd
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
import ast
import copy
import datetime
# Import reactive approach
from scheduling_methods.reactive_method import run_reactive_online

# Import proactive approach
from scheduling_methods.proactive_method import run_proactive_offline, run_proactive_online

# Import STNU approach
from scheduling_methods.stnu_method import run_stnu_offline, run_stnu_online

from general.logger import get_logger
logger = get_logger(__name__)

"""
This script contains the experiments that are presented in the AAAI'25 submission "Proactive and Reactive
Constraint Programming for Stochastic Project Scheduling with Maximal Time-Lags". Note that while running 
these experiments a new csv file is generated including the results with a time stamp in the filename

"""

# GENERAL SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'

perfect_information = False

# SETTINGS HEURISTIC PROACTIVE APPROACH
mode_proactive = "quantile_0.9"
time_limit_proactive = 600
# SETTINGS REACTIVE APPROACH
time_limit_rescheduling = 2
# SETTINGS SAA APPROACH
mode_saa = "SAA_smart"
time_limit_saa = 1800
nb_scenarios_saa = 4
# SETTINGS STNU APPROACH
time_limit_cp_stnu = 600
mode_stnu = "robust"


# SETTINGS EXPERIMENTS
DIRECTORY_PI = "aaai25/results_perfect_information"
INSTANCE_FOLDERS = ["j10", "j20", "j30", "ubo50", "ubo100"]
INSTANCE_IDS = range(1, 51)
NOISE_FACTORS = [1, 2]
nb_scenarios_test = 10
proactive_reactive = True
proactive_saa = True
stnu = True
writing = False
now = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")


def check_pi_feasible(instance_folder, instance_id, sample_index, duration_sample, noise_factor):
    df = pd.read_csv(f'{DIRECTORY_PI}/results_pi_{instance_folder}_{noise_factor}.csv')
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


for noise_factor in NOISE_FACTORS:
    output_file = f'final_results_{noise_factor}_{now}.csv'
    data = []
    for instance_folder in INSTANCE_FOLDERS:
        for instance_id in INSTANCE_IDS:

            # PREPARE DATA AND DURATION SAMPLES
            rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor)
            np.random.seed(SEED)
            test_durations_samples = rcpsp_max.sample_durations(nb_scenarios_test)

            # RUN PROACTIVE / REACTIVE OFFLINE
            data_dict_baseline = run_proactive_offline(rcpsp_max, time_limit_proactive, mode_proactive, nb_scenarios_saa)

            if proactive_reactive:
                # RUN PROACTIVE ONLINE
                data_dict_proactive = copy.copy(data_dict_baseline)
                for i, duration_sample in enumerate(test_durations_samples):
                    pi_feasible, obj_pi = check_pi_feasible(instance_folder, instance_id, i, duration_sample, noise_factor)
                    data_dict_proactive["obj_pi"] = obj_pi
                    if pi_feasible:
                        data += run_proactive_online(rcpsp_max, duration_sample, data_dict_proactive)
                        data_df = pd.DataFrame(data)
                        if writing:
                            data_df.to_csv(output_file, index=False)
                    else:
                        logger.info(f'Instance {rcpsp_max.instance_folder}PSP{rcpsp_max.instance_id}, sample {i}: '
                                    f'We can skip the proactive approach')

                # RUN REACTIVE ONLINE
                data_dict_reactive = copy.copy(data_dict_baseline)
                data_dict_reactive["method"] = "reactive"
                data_dict_reactive["estimated_start_times"] = data_dict_reactive["start_times"]

                for i, duration_sample in enumerate(test_durations_samples):
                    pi_feasible, obj_pi = check_pi_feasible(instance_folder, instance_id, i, duration_sample, noise_factor)
                    if pi_feasible:
                        data_dict_reactive["obj_pi"] = obj_pi
                        data += run_reactive_online(rcpsp_max, duration_sample, data_dict_reactive, time_limit_rescheduling)
                        data_df = pd.DataFrame(data)
                        if writing:
                            data_df.to_csv(output_file, index=False)
                    else:
                        logger.info(
                            f'Instance {rcpsp_max.instance_folder}PSP{rcpsp_max.instance_id}, sample {i}: We can skip the reactive approach')

            # RUN PROACTIVE SAA OFFLINE
            if proactive_saa:
                data_dict_saa = run_proactive_offline(rcpsp_max, time_limit_saa, mode_saa, nb_scenarios_saa)

                # RUN PROACTIVE SAA ONLINE
                for i, duration_sample in enumerate(test_durations_samples):
                    pi_feasible, obj_pi = check_pi_feasible(instance_folder, instance_id, i, duration_sample, noise_factor)
                    data_dict_saa["obj_pi"] = obj_pi
                    if pi_feasible:
                        data += run_proactive_online(rcpsp_max, duration_sample, data_dict_saa)
                        data_df = pd.DataFrame(data)
                        if writing:
                            data_df.to_csv(output_file, index=False)
                    else:
                        logger.info(f'Instance {rcpsp_max.instance_folder}PSP{rcpsp_max.instance_id}, sample {i}: '
                                    f'We can skip the proactive approach')

            # RUN STNU EXPERIMENTS
            if stnu:
                rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id,
                                                            noise_factor)
                dc, estnu, data_dict_stnu = run_stnu_offline(rcpsp_max, time_limit_cp_stnu=time_limit_cp_stnu, mode=mode_stnu)

                for i, duration_sample in enumerate(test_durations_samples):
                    pi_feasible, obj_pi = check_pi_feasible(instance_folder, instance_id, i, duration_sample, noise_factor)
                    data_dict_stnu["obj_pi"] = obj_pi
                    if pi_feasible:
                        data += run_stnu_online(dc, estnu, duration_sample, rcpsp_max, data_dict_stnu)
                        data_df = pd.DataFrame(data)
                        if writing:
                            data_df.to_csv(output_file, index=False)
                    else:
                        logger.info(f'Instance {rcpsp_max.instance_folder}PSP{rcpsp_max.instance_id}, sample {i}:'
                                    f' We can skip the STNU approach')


