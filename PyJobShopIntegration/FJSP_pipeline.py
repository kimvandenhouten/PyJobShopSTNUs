import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyjobshop.plot import plot_machine_gantt

import general.logger
from datetime import datetime

from FJSP import FJSP
from proactive import run_proactive_offline, run_proactive_online_direct
from reactive import run_reactive_online, run_reactive_offline
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
dir_path = Path(__file__).resolve().parent
logger = general.logger.get_logger(__name__)

# Configuration
DATA_ROOT = os.path.join(dir_path, "data", "fjsp_sdst", "fattahi")
IMAGES_ROOT = os.path.join(dir_path, "images", "fjsp_sdst")
PROBLEM_TYPE = "fjsp_sdst"
NOISE_FACTORS = [1.0, 2.0]
stnu_time_limit = 10000
proactive_time_limit = 10000
reactive_offline_time_limit = 10000
time_limit_rescheduling = 7
proactive_mode = ['robust', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.9']
reactive_mode = ['mean', 'robust', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.9']
number_samples = 50
methods = ('proactive', 'reactive')

# Timestamp for results
now = datetime.now().strftime("%m_%d_%Y,%H_%M")
# CSV output for evaluator
results_dir = get_project_root() / "PyJobShopIntegration" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
output_file = results_dir / f"final_results_{now}.csv"


all_files = sorted(os.listdir(DATA_ROOT))

for noise in NOISE_FACTORS:
    for file_name in all_files:
        instance_name = os.path.splitext(file_name)[0]

        # Paths
        instance_path = os.path.join(DATA_ROOT, file_name)
        out_folder = os.path.join(IMAGES_ROOT, instance_name, f"noise_{noise}")
        os.makedirs(out_folder, exist_ok=True)

        logger.info(f"Processing {file_name} with noise factor {noise}")
        instance_name = os.path.splitext(file_name)[0]

        # Paths
        instance_path = os.path.join(DATA_ROOT, file_name)
        out_folder = os.path.join(IMAGES_ROOT, instance_name, f"noise_{noise}")
        os.makedirs(out_folder, exist_ok=True)

        logger.info(f"Processing {file_name} with noise factor {noise}")
        if 'proactive' in methods:
            # --- Offline phase: CP solver ---
            logger.info(f"---------------------PROACTIVE APPROACH---------------------")
            model = create_instance(instance_path, PROBLEM_TYPE, PROBLEM_TYPE == "fjsp_sdst")
            fjsp_instance = FJSP(model)
            for pmode in proactive_mode:
                data_dict_offline, result = run_proactive_offline(fjsp_instance, noise, proactive_time_limit, pmode)
                logger.info(f"CP deterministic makespan for {instance_name} noise {noise}: {result.objective}")
                # --- Online phase: sampling + execution ---
                real_durations = fjsp_instance.duration_distributions(noise_factor=noise).sample(number_samples)
                real_durations = np.atleast_2d(real_durations)
                # --- Online phase: sampling + execution ---
                for i, duration_sample in enumerate(real_durations):
                    data_dict_proactive = run_proactive_online_direct(duration_sample=duration_sample, data_dict=data_dict_offline, fjsp_instance=fjsp_instance, result=result)
                    data_to_csv(instance_folder=instance_name, solution=data_dict_proactive, output_file=output_file)
                    if data_dict_proactive["feasibility"]:
                        logger.info(f"Simulated makespan for {instance_name} noise {noise} sample {i}: {data_dict_proactive['obj']}")
                    else:
                        logger.info(f"Simulation for {instance_name} noise {noise} sample {i} is infeasible")
        if 'reactive' in methods:
            # --- Offline phase: CP solver ---
            logger.info(f"---------------------REACTIVE APPROACH---------------------")
            model = create_instance(instance_path, PROBLEM_TYPE, PROBLEM_TYPE == "fjsp_sdst")
            for rmode in reactive_mode:
                fjsp_instance = FJSP(model)
                data_dict_offline_reactive, result = run_reactive_offline(fjsp_instance, noise, reactive_offline_time_limit, rmode)
                logger.info(f"CP deterministic makespan for {instance_name} noise {noise}: {result.objective}")
                # --- Online phase: sampling + execution ---
                real_durations = fjsp_instance.duration_distributions(noise_factor=noise).sample(number_samples)
                real_durations = np.atleast_2d(real_durations)
                # --- Online phase: sampling + execution ---
                for i, duration_sample in enumerate(real_durations):
                    data_dict_proactive = run_reactive_online(duration_sample=duration_sample,
                                                                      data_dict=data_dict_offline_reactive,
                                                                      fjsp_instance=fjsp_instance, result=result, time_limit_rescheduling=time_limit_rescheduling)
                    data_to_csv(instance_folder=instance_name, solution=data_dict_proactive, output_file=output_file)
                    if data_dict_proactive["feasibility"]:
                        logger.info(
                            f"Simulated makespan for {instance_name} noise {noise} sample {i}: {data_dict_proactive['obj']}")
                    else:
                        logger.info(f"Simulation for {instance_name} noise {noise} sample {i} is infeasible")
        if 'stnu' in methods:
            logger.info(f"---------------------STNU APPROACH---------------------")
            # --- Offline phase: CP solver ---
            model = create_instance(instance_path, PROBLEM_TYPE, PROBLEM_TYPE=="fjsp_sdst")
            start_offline = time.time()
            result = model.solve(solver='cpoptimizer', display=False, time_limit=stnu_time_limit)
            finish_offline = time.time()
            solution = result.best
            logger.info(f"CP deterministic makespan for {instance_name} noise {noise}: {result.objective}")
            fjsp = FJSP(model)
            duration_distributions = fjsp.duration_distributions(noise_factor=noise)
            stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions, True, solution.tasks)
            stnu.add_resource_chains(solution, model)

            # Export XML for DC check
            xml_base = f"{instance_name}_noise{noise}"
            xml_dir = os.path.join(dir_path, "temporal_networks", "cstnu_tool", "xml_files")
            os.makedirs(xml_dir, exist_ok=True)
            stnu_to_xml(stnu, xml_base, xml_dir)

            # DC check
            dc, xml_out = run_dc_algorithm(xml_dir, xml_base)
            logger.debug(f"DC result for {instance_name} noise {noise}: {dc}")

            # If not DC, record infeasible and skip reactive execution
            if not dc:
                solution_record = {
                    'method': 'stnu',
                    'instance_folder': instance_name,
                    'noise_factor': noise,
                    'time_offline': finish_offline - start_offline,
                    'time_online': 0.0,
                    'obj': float('nan'),
                    'feasibility': False,
                    'start_times': {},
                    'time_limit': stnu_time_limit,
                    'real_durations': {}

                }
                data_to_csv(instance_folder=instance_name, solution=solution_record, output_file=str(output_file))
                continue

            real_durations = duration_distributions.sample(number_samples)
            real_durations = np.atleast_2d(real_durations)
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
                    'feasibility': True,
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