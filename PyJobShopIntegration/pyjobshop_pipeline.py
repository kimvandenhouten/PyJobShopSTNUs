import sys
import os
import datetime
import numpy as np
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from PyJobShopIntegration.utils import add_resource_chains, get_resource_chains, sample_for_rte
from general.logger import get_logger
from parser import parse_data
from evaluator import evaluate_results
from scheduling_methods.stnu_method import get_start_and_finish
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU

logger = get_logger(__name__)

problem_type = sys.argv[-1]
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
INSTANCE_FOLDERS = ["j10", "j20", "j30", "ubo50", "ubo100"]
INSTANCE_IDS = range(1, 51)
NOISE_FACTORS = [1, 2]
nb_scenarios_test = 10
proactive_reactive = True
proactive_saa = True
stnu = True
writing = False
now = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")

for noise_factor in NOISE_FACTORS:
    output_file = f'final_results_{noise_factor}_{now}.csv'
    data = []
    for instance_folder in INSTANCE_FOLDERS:
        folder_path = os.path.join("data", folder, instance_folder)
        for file in os.listdir(folder_path):
            # Load data
            instance = parse_data(os.path.join(folder_path, file), problem_type)
            test_durations_samples = instance.sample_durations(nb_scenarios_test)
            # Run experiments on proactive, reactive and stnu
            # TODO implement the proactive, reactive and stnu approaches possibly reusing already existing code
            for i, duration_sample in enumerate(test_durations_samples):
                if proactive_reactive:
                    pass
                if proactive_saa:
                    pass
                if stnu:
                    model = instance.create_model(duration_sample)
                    result = model.solve(time_limit=5, display=False)
                    duration_distributions = DiscreteUniformSampler(
                        lower_bounds=np.random.randint(1, 3, len(model.tasks)),
                        upper_bounds=np.random.randint(5, 8, len(model.tasks)))
                    stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
                    result_tasks = result.best.tasks
                    # TODO potentially add other fields depending on the problem
                    schedule = [{"task": i, "start": task.start, "end": task.end, "mode": task.mode}
                                for i, task in enumerate(result_tasks)]
                    needs = []
                    # TODO this might not work if mode is None
                    for task in result_tasks:
                        needs.append(task.mode.needs)
                    resource_chains, resource_assignments = get_resource_chains(
                        schedule, instance.capacity, needs, complete=True)
                    stnu = add_resource_chains(stnu, resource_chains)
                    file_name = f"{problem_type}_pyjobshop_stnu_{file}_{noise_factor}_{i}"
                    stnu.to_xml(file_name, "temporal_networks/cstnu_tool/xml_files")
                    dc, output_location = run_dc_algorithm(
                        "temporal_networks/cstnu_tool/xml_files", file_name)
                    if dc:
                        logger.info(f'The network resulting from the PyJobShop solution is DC')
                        estnu = STNU.from_graphml(output_location)
                        rte_sample = sample_for_rte(duration_sample, estnu)
                        logger.debug(f"Sample dict that will be given to RTE star is {rte_sample}")
                        rte_data = rte_star(estnu, oracle="sample", sample=rte_sample)
                        start_times, finish_times = get_start_and_finish(estnu, rte_data, instance.num_tasks)
                        solution = {}
                        solution['obj'] = instance.get_objective(rte_data, objective="makespan")
                        solution['feasibility'] = instance.check_feasibility()
                        solution['start_times'] = start_times
                        solution['time_online'] = finish_online - start_online
                    else:
                        logger.info(f'The network is not DC')
    # Analyze the results perform statistical tests and create plots
    evaluate_results(data)

