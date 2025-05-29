import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
from pyjobshop import SolveStatus
from pyjobshop.plot import plot_machine_gantt

import general.logger
from datetime import datetime

from FJSP_Mayte.FJSP import FJSP
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.parser import create_instance
from PyJobShopIntegration.utils import (
    sample_for_rte,
    rte_data_to_pyjobshop_solution,
    data_to_csv,
    get_project_root, get_start_and_finish_from_rte
)
from evaluator import evaluate_results
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU

# Initialize logger and paths
dir_path = Path(__file__).resolve().parent.parent
logger = general.logger.get_logger(__name__)

# Configuration
DATA_ROOT = os.path.join(dir_path, "data", "fjsp_sdst", "fattahi")
IMAGES_ROOT = os.path.join(dir_path, "images", "fjsp_sdst")
PROBLEM_TYPE = "fjsp_sdst"
NOISE_FACTORS = [1.0, 2.0]
stnu_time_limit = 10000000
number_samples = 10

# Timestamp for results
now = datetime.now().strftime("%m_%d_%Y,%H_%M")
# CSV output for evaluator
results_dir = get_project_root() / "PyJobShopIntegration" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
output_file = results_dir / f"final_results_{now}.csv"

# Batch execution
for noise in NOISE_FACTORS:
    for file_name in os.listdir(DATA_ROOT):
        instance_name = os.path.splitext(file_name)[0]
        if (instance_name in ("Fattahi_setup_19",  "Fattahi_setup_20")):
            continue

        # Paths
        instance_path = os.path.join(DATA_ROOT, file_name)
        out_folder = os.path.join(IMAGES_ROOT, instance_name, f"noise_{noise}")
        os.makedirs(out_folder, exist_ok=True)

        logger.info(f"Processing {file_name} with noise factor {noise}")

        # --- Offline phase: CP solver ---
        start_offline = time.time()
        model = create_instance(instance_path, PROBLEM_TYPE, PROBLEM_TYPE=="fjsp_sdst")
        result = model.solve(solver='cpoptimizer', display=False, time_limit=stnu_time_limit)
        solution = result.best
        if result.status == SolveStatus.OPTIMAL:
            logger.info(f"Solution for {file_name} with noise factor {noise} is optimal")
        finish_offline = time.time()

        fjsp = FJSP(model)
        # Define the stochastic processing time distributions
        duration_distributions = fjsp.duration_distributions(noise_factor=noise)
        # Construct STNU and add chains
        stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
        stnu.add_resource_chains(solution, model)

        # Export XML for DC check
        xml_base = f"{instance_name}_noise{noise}"
        xml_dir = os.path.join(dir_path, "temporal_networks", "cstnu_tool", "xml_files")
        os.makedirs(xml_dir, exist_ok=True)
        stnu_to_xml(stnu, xml_base, xml_dir)

        # DC check
        dc, xml_out = run_dc_algorithm(xml_dir, xml_base)
        logger.info(f"DC result for {instance_name} noise {noise}: {dc}")

        # If not DC, record infeasible and skip reactive execution
        if not dc:
            solution_record = {
                'method': 'stnu',
                'instance_folder': instance_name,
                'noise_factor': noise,
                'time_offline': finish_offline - start_offline,
                'time_online': 0.0,
                'obj': float('nan'),
                'feasibility': 0,
                'start_times': {},
                'time_limit': stnu_time_limit,
                'real_durations': {}

            }
            data_to_csv(instance_folder=instance_name, solution=solution_record, output_file=str(output_file))
            continue

        real_durations = duration_distributions.sample(number_samples)
        # --- Online phase: sampling + execution ---
        for i, duration_sample in enumerate(real_durations):
            start_online = time.time()
            estnu = STNU.from_graphml(xml_out)
            real_durations = duration_distributions.sample()
            sample = sample_for_rte(real_durations, estnu)
            rte_data = rte_star(estnu, oracle='sample', sample=sample)
            finish_online = time.time()

            # Map back to a PyJobShop solution
            start_times, finish_times = get_start_and_finish_from_rte(estnu, rte_data, len(model.tasks))
            simulated_solution, makespan = rte_data_to_pyjobshop_solution(
                solution, estnu, rte_data, len(model.tasks), "makespan"
            )
            logger.info(f"Simulated makespan for {instance_name} noise {noise} sample {i}: {makespan} ")

            # Record results for evaluator
            solution_record = {
                'method': 'stnu',
                'instance_folder': instance_name,
                'noise_factor': noise,
                'time_offline': finish_offline - start_offline,
                'time_online': finish_online - start_online,
                'obj': makespan,
                'feasibility': 1,
                'start_times': start_times,
                'time_limit': stnu_time_limit,
                'real_durations': str(real_durations)
            }

            data_to_csv(instance_folder=instance_name, solution=solution_record, output_file=str(output_file))

            # Plot and save Gantt chart
            plot_machine_gantt(simulated_solution, model.data(), plot_labels=True)
            out_png = os.path.join(out_folder, f"{instance_name}_gantt_noise{noise}_sample{i}.png")
            plt.savefig(out_png)
            plt.close()

# After batch: run evaluation report
evaluate_results(now)