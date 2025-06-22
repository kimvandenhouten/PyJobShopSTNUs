import copy
import random
import sys
import os
import datetime
import time

import numpy as np
from matplotlib import pyplot as plt
from pyjobshop import Solution, TaskData
from pyjobshop.plot import plot_task_gantt, plot_resource_usage

from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from PyJobShopIntegration.utils import add_resource_chains, get_resource_chains, sample_for_rte, plot_stnu, data_to_csv
from general.logger import get_logger
from PyJobShopIntegration.evaluator import evaluate_results
from PyJobShopIntegration.parser import create_instance
from scheduling_methods.stnu_method import get_start_and_finish
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU
from PyJobShopIntegration.scheduling_methods.proactive_method import run_proactive_offline, run_proactive_online
from PyJobShopIntegration.scheduling_methods.reactive_method import run_reactive_online

random.seed(42)
np.random.seed(45)
logger = get_logger(__name__)
# the problem type is passed as a command line argument e.g. python pyjobshop_pipeline.py mmrcpspd
problem_type = sys.argv[-1]
# make sure to have a folder with your data with the same name
folder = problem_type
# SETTINGS HEURISTIC PROACTIVE APPROACH
mode_proactive = "robust"  # can be "robust", "quantile_0.25", "quantile_0.5", "quantile_0.75", "quantile_0.9"
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
multimode = problem_type.startswith("mm")

# SETTINGS EXPERIMENTS
INSTANCE_FOLDERS = ["j10", "j20"]
NOISE_FACTORS = [1, 2]
nb_scenarios_test = 10
proactive_reactive = False
proactive_saa = True
stnu = True
writing = False
now = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")
path = os.path.join(os.getcwd(), "PyJobShopIntegration")
infeasible_sample = {}
infeasible_sample["stnu"] = {}
infeasible_sample["proactive"] = {}
infeasible_sample["reactive"] = {}
t = str(time.time())
for noise_factor in NOISE_FACTORS:
    infeasible_sample["stnu"][noise_factor] = {}
    output_file = f'final_results_{now}.csv'
    for j, instance_folder in enumerate(INSTANCE_FOLDERS):
        folder_path = os.path.join(path, "data", folder, instance_folder)
        # create a folder in images for results of this experiment
        images_folder = os.path.join(path, "images", problem_type, t, f"noise_factor{noise_factor}", instance_folder)
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        for n, file in enumerate(os.listdir(folder_path)):
            logger.info(f"---------------------------------{file}---------------------------------")
            if not os.path.exists(os.path.join(images_folder, file)):
                os.makedirs(os.path.join(images_folder, file))
            # Keep it short for testing
            print(n)
            if n == 100:
                break
            # Load data
            instance = create_instance(os.path.join(folder_path, file), problem_type)
            test_durations_samples, duration_distributions = instance.sample_durations(nb_scenarios_test, noise_factor)
            # Run experiments on proactive, reactive and stnu
            # TODO implement the proactive, reactive and stnu approaches possibly reusing already existing code
            for i, duration_sample in enumerate(test_durations_samples):
                logger.info(f"Sample {i}")
                # instance.get_deterministic_makespan()
                if proactive_reactive:
                    data_dict = run_proactive_offline(instance, noise_factor, time_limit_proactive, mode_proactive)
                    # Run proactive online
                    data_dict_proactive = copy.copy(data_dict)
                    result_tasks = data_dict["result_tasks"]
                    real_durations = instance.get_real_durations(result_tasks, duration_sample)
                    lb, ub = instance.get_bounds(noise_factor=noise_factor)
                    # print warning if any duration sample is higher than ub and which element is higher
                    for k, duration in enumerate(duration_sample):
                        if duration > ub[k]:
                            print(duration_sample)
                            print(ub)
                            raise ValueError(f"Duration sample {duration} is higher than upper bound {ub[k]} for task {k}")
                        if duration < lb[k]:
                            print(duration_sample)
                            print(lb)
                            raise ValueError(f"Duration sample {duration} is lower than lower bound {lb[k]} for task {k}")
                    if real_durations == []:
                        logger.info("The solution is infeasible")
                    else:
                        data_dict_proactive = run_proactive_online(instance, real_durations, data_dict_proactive)
                        data_to_csv(instance_folder=instance_folder, solution=data_dict_proactive, output_file=output_file)

                        # Run reactive online
                        data_dict_reactive = copy.copy(data_dict)
                        data_dict_reactive = run_reactive_online(instance, real_durations, data_dict_reactive, time_limit_rescheduling, result_tasks)
                        data_dict_reactive["method"] = "reactive"
                        data_to_csv(instance_folder=instance_folder, solution=data_dict_reactive, output_file=output_file)
                if proactive_saa:
                    pass
                if stnu:
                    start_offline = time.time()
                    model = instance.create_model(instance.sample_mode(mode_stnu, noise_factor))
                    result = model.solve(time_limit=5, display=False, random_seed=42, num_search_workers=1)
                    result_tasks = result.best.tasks
                    if result_tasks == []:
                        # print(f"Infeasible solution for duration sample: {duration_sample}, file: {file}, noise factor: {noise_factor}")
                        logger.info("The solution is infeasible")
                        continue
                    stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions=duration_distributions, result_tasks=result_tasks, multimode=multimode)
                    # TODO potentially add other fields depending on the problem
                    schedule = instance.get_schedule(result_tasks)
                    real_durations = instance.get_real_durations(result_tasks, duration_sample)
                    # TODO the update of the infeasible sample needs to happen elsewhere
                    if real_durations == []:
                        if file not in infeasible_sample["stnu"][noise_factor]:
                            infeasible_sample["stnu"][noise_factor][file] = 0
                        infeasible_sample["stnu"][noise_factor][file] += 1
                        logger.info("The solution is infeasible")
                        continue
                    demands = []
                    # TODO this might not work if mode is None
                    for i, task in enumerate(result_tasks):
                        mode = task.mode
                        demands.append(instance.modes[mode].demands)
                    resource_chains, resource_assignments = get_resource_chains(
                        schedule, instance.capacities, demands, complete=True)
                    stnu = add_resource_chains(stnu, resource_chains)
                    file_name = f"{problem_type}_pyjobshop_stnu_{file}_{noise_factor}_{i}"
                    stnu_to_xml(stnu, file_name, "temporal_networks/cstnu_tool/xml_files")
                    dc, output_location = run_dc_algorithm(
                        "temporal_networks/cstnu_tool/xml_files", file_name)
                    if dc:
                        # logger.info(f'The network resulting from the PyJobShop solution is DC for sample {ds}')
                        estnu = STNU.from_graphml(output_location)
                        rte_sample = sample_for_rte(real_durations, estnu)
                        finish_offline = time.time()
                        start_online = time.time()
                        rte_data = rte_star(estnu, oracle="sample", sample=rte_sample)
                        if type(rte_data) == bool:
                            logger.info("The solution is infeasible")
                            continue
                        start_times, finish_times = get_start_and_finish(estnu, rte_data, len(model.tasks))
                        finish_online = time.time()
                        demands = []
                        for i, task in enumerate(result_tasks):
                            mode = task.mode
                            demands.append(instance.modes[mode].demands)
                        feasibility = instance.check_feasibility(start_times, finish_times, real_durations, demands)
                        if not feasibility:
                            finish_offline = time.time()
                            solution = {'obj': np.inf,
                                        'feasibility': False,
                                        'start_times': [],
                                        'time_online': np.inf,
                                        'time_offline': finish_offline - start_offline,
                                        'noise_factor': noise_factor,
                                        'method': 'stnu',
                                        'time_limit': time_limit_cp_stnu,
                                        'real_durations': str(real_durations)}
                            data_to_csv(instance_folder=instance_folder, solution=solution, output_file=output_file)
                            logger.info(f'The network is not DC for sample{real_durations}')
                            continue
                        schedule = []
                        for i, (start, end) in enumerate(zip(start_times, finish_times)):
                            if i < instance.num_tasks - 1:
                                schedule.append({
                                    'task': i,
                                    'start': start,
                                    'end': end,
                                })
                            else:
                                schedule.append({
                                    'task': i,
                                    'start': 0,
                                    'finish': end-start,
                                })
                        solution = {'obj': instance.get_objective(schedule, objective="makespan"),
                                    'feasibility': feasibility,
                                    'start_times': start_times,
                                    'time_online': finish_online - start_online,
                                    'time_offline': finish_offline - start_offline,
                                    'noise_factor': noise_factor,
                                    'method': 'stnu',
                                    'time_limit': time_limit_cp_stnu,
                                    'real_durations': str(real_durations)}
                        task_data = []
                        for task, start, finish in zip(result_tasks, start_times, finish_times):
                            mode = task.mode
                            resources = task.resources
                            task_data.append(TaskData(mode, resources, start, finish))
                        solution_plot = Solution(task_data)
                        d = model.data()
                        fig, axes = plt.subplots(
                            d.num_resources + 1,
                            figsize=(12, 16),
                            gridspec_kw={"height_ratios": [6] + [1] * d.num_resources},
                            )
                        try:
                            plot_task_gantt(solution_plot, d, ax=axes[0])
                            plot_resource_usage(solution_plot, d, axes=axes[1:])
                            time_now = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")
                            plt.savefig(os.path.join(os.path.join(images_folder, file), f'{file}_{i}_{noise_factor}_{time_now}.png'))
                            plt.close(fig)
                        except Exception as e:
                            logger.error(f"Error plotting Gantt chart: {e}")
                        data_to_csv(instance_folder=instance_folder, solution=solution, output_file=output_file)
                    # TODO update infeasible solutions count
                    else:
                        finish_offline = time.time()
                        solution = {'obj': np.inf,
                                    'feasibility': False,
                                    'start_times': [],
                                    'time_online': np.inf,
                                    'time_offline': finish_offline - start_offline,
                                    'noise_factor': noise_factor,
                                    'method': 'stnu',
                                    'time_limit': time_limit_cp_stnu,
                                    'real_durations': str(real_durations)}
                        data_to_csv(instance_folder=instance_folder, solution=solution, output_file=output_file)
                        logger.info(f'The network is not DC for sample{real_durations}')
    # Analyze the results perform statistical tests and create plots
evaluate_results(now=now)
# TODO potentially add this to evaluate_results for analysis
print(f"Number of infeasible samples: {infeasible_sample}")

