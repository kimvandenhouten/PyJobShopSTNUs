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
from PyJobShopIntegration.utils import add_resource_chains, get_resource_chains, sample_for_rte, plot_stnu
from general.logger import get_logger
from PyJobShopIntegration.evaluator import evaluate_results
from PyJobShopIntegration.parser import create_instance
from scheduling_methods.stnu_method import get_start_and_finish
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU

logger = get_logger(__name__)
# the problem type is passed as a command line argument e.g. python pyjobshop_pipeline.py mmrcpspd
problem_type = sys.argv[-1]
# make sure to have a folder with your data with the same name
folder = problem_type
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
DIRECTORY_PI = os.path.join("aaai25", "results_perfect_information")
INSTANCE_FOLDERS = ["j10"]
NOISE_FACTORS = [1, 2]
nb_scenarios_test = 10
proactive_reactive = True
proactive_saa = True
stnu = True
writing = False
now = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")
path = os.path.join(os.getcwd(), "PyJobShopIntegration")
infeasible_sample = {}
infeasible_sample["stnu"] = {}
infeasible_sample["proactive"] = {}
infeasible_sample["reactive"] = {}
for noise_factor in NOISE_FACTORS:
    infeasible_sample["stnu"][noise_factor] = {}
    output_file = f'final_results_{noise_factor}_{now}.csv'
    data = []
    for j, instance_folder in enumerate(INSTANCE_FOLDERS):
        folder_path = os.path.join(path, "data", folder, instance_folder)
        # create a folder in images for results of this experiment
        images_folder = os.path.join(path, "images", problem_type, str(time.time()), f"noise_factor{noise_factor}", instance_folder)
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        for n, file in enumerate(os.listdir(folder_path)):
            if not os.path.exists(os.path.join(images_folder, file)):
                os.makedirs(os.path.join(images_folder, file))
            # Keep it short for testing
            if n == 100:
                break
            # Load data
            instance = create_instance(os.path.join(folder_path, file), problem_type)
            test_durations_samples, duration_distributions = instance.sample_durations(nb_scenarios_test, noise_factor)
            # Run experiments on proactive, reactive and stnu
            # TODO implement the proactive, reactive and stnu approaches possibly reusing already existing code
            for i, duration_sample in enumerate(test_durations_samples):
                if proactive_reactive:
                    pass
                if proactive_saa:
                    pass
                if stnu:
                    start_online = time.time()
                    model = instance.create_model(duration_sample)
                    result = model.solve(time_limit=5, display=False)
                    result_tasks = result.best.tasks
                    if result_tasks == []:
                        print(f"Infeasible solution for duration sample: {duration_sample}, file: {file}, noise factor: {noise_factor}")
                        logger.info("The solution is infeasible")
                        continue
                    stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions=duration_distributions, result_tasks=result_tasks)
                    # TODO potentially add other fields depending on the problem
                    schedule = instance.get_schedule(result_tasks)
                    duration_sample = [instance.modes[task.mode].duration for task in result_tasks]
                    # TODO the update of the infeasible sample needs to happen elsewhere
                    if duration_sample == []:
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
                        logger.info(f'The network resulting from the PyJobShop solution is DC for sample {duration_sample}')
                        estnu = STNU.from_graphml(output_location)
                        rte_sample = sample_for_rte(duration_sample, estnu)
                        rte_data = rte_star(estnu, oracle="sample", sample=rte_sample)
                        if type(rte_data) == bool:
                            logger.info("The solution is infeasible")
                            continue
                        start_times, finish_times = get_start_and_finish(estnu, rte_data, len(model.tasks))
                        finish_online = time.time()
                        solution = {'obj': instance.get_objective(rte_data, objective="makespan"),
                                    'feasibility': instance.check_feasibility(start_times, finish_times),
                                    'start_times': start_times,
                                    'time_online': finish_online - start_online,
                                    'real_durations': duration_sample}
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

                        plot_task_gantt(solution_plot, d, ax=axes[0])
                        plot_resource_usage(solution_plot, d, axes=axes[1:])
                        plt.savefig(os.path.join(os.path.join(images_folder, file), f'{file}_{i}_{noise_factor}_{time.time()}.png'))
                        data.append(solution)
                        if not solution['feasibility']:
                            print("Finish times: ", finish_times)
                            raise ValueError("The solution is infeasible")
                    else:
                        logger.info(f'The network is not DC for sample{duration_sample}')
    print("Data: ", data)
    # Analyze the results perform statistical tests and create plots
    evaluate_results(data)
# TODO potentially add this to evaluate_results for analysis
print(f"Number of infeasible samples: {infeasible_sample}")

