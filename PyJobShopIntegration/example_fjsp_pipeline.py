import argparse
import os
import sys
import time
import datetime
import logging

import numpy as np
from matplotlib import pyplot as plt

from pyjobshop import Solution, TaskData
from pyjobshop.plot import plot_task_gantt, plot_resource_usage

from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from PyJobShopIntegration.utils import (
    add_resource_chains,
    get_resource_chains,
    sample_for_rte,
    plot_stnu,
    data_to_csv, rte_data_to_pyjobshop_solution,
)
from PyJobShopIntegration.evaluator import evaluate_results
from PyJobShopIntegration.parser import create_instance
from scheduling_methods.stnu_method import get_start_and_finish
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU
from PyJobShopIntegration.scheduling_methods.proactive_method import (
    run_proactive_offline,
    run_proactive_online,
)
from PyJobShopIntegration.scheduling_methods.reactive_method import run_reactive_online

# Configuration settings
CONFIG = {
    # Heuristic proactive approach
    "proactive": {
        "mode": "robust",
        "time_limit_offline": 600,
    },
    # Reactive rescheduling
    "reactive": {
        "time_limit_rescheduling": 2,
    },
    # SAA approach
    "saa": {
        "mode": "SAA_smart",
        "time_limit": 1800,
        "nb_scenarios": 4,
    },
    # STNU approach
    "stnu": {
        "mode": "robust",
        "time_limit_cp": 600,
    },
    # Experiment parameters
    "experiments": {
        "instance_folders": ["fattahi"],
        "noise_factors": [1],
        "nb_scenarios_test": 10,
        "proactive_reactive": False,
        "proactive_saa": False,
        "stnu": True,
        "writing": True,
    },
    # Paths
    "paths": {
        "base": os.path.join(os.getcwd(), "PyJobShopIntegration"),
        "image_subdir": "images",
        "xml_dir": "temporal_networks/cstnu_tool/xml_files",
    },
}


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyJobShop scheduling pipeline"
    )
    parser.add_argument(
        "problem_type",
        help="Problem type, e.g. 'mmrcpspd'",
    )
    return parser.parse_args()


def create_output_dirs(folder_path, images_folder, file_name):
    os.makedirs(images_folder, exist_ok=True)
    instance_img_dir = os.path.join(images_folder, file_name)
    os.makedirs(instance_img_dir, exist_ok=True)
    return instance_img_dir


def run_proactive_and_reactive(instance, duration_sample, config, output_file, instance_folder):
    # Proactive offline
    data_dict = run_proactive_offline(
        instance,
        config['proactive']['mode'],
        config['proactive']['time_limit_offline'],
    )
    # Real durations and bounds check
    result_tasks = data_dict['result_tasks']
    real_durations = instance.get_real_durations(result_tasks, duration_sample)
    lb, ub = instance.get_bounds(noise_factor=config['noise_factor'])
    for idx, duration in enumerate(duration_sample):
        if duration > ub[idx] or duration < lb[idx]:
            raise ValueError(
                f"Duration sample {duration} out of bounds [{lb[idx]}, {ub[idx]}] for task {idx}"
            )
    if real_durations:
        # Proactive online
        proactive_data = run_proactive_online(instance, real_durations, data_dict.copy())
        data_to_csv(instance_folder, proactive_data, output_file)
        # Reactive online
        reactive_data = run_reactive_online(
            instance,
            real_durations,
            data_dict.copy(),
            config['reactive']['time_limit_rescheduling'],
            result_tasks,
        )
        reactive_data['method'] = 'reactive'
        data_to_csv(instance_folder, reactive_data, output_file)


def run_stnu_pipeline(
    model, instance, duration_distributions, result_tasks,
    real_durations, schedule, noise_factor, images_folder, instance_folder,
):
    stnu = PyJobShopSTNU.from_concrete_model(
        model, duration_distributions, result_tasks, multimode=instance.multimode
    )
    demands = [instance.modes[t.mode].demands for t in result_tasks]
    chains, assignments = get_resource_chains(
        schedule, instance.capacities, demands, complete=True
    )
    stnu = add_resource_chains(stnu, chains)

    # Export XML and run DC
    timestamp = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")
    base_name = f"{instance_folder}_stnu_{noise_factor}_{timestamp}"
    stnu_to_xml(stnu, base_name, CONFIG['paths']['xml_dir'])
    dc_ok, out_loc = run_dc_algorithm(CONFIG['paths']['xml_dir'], base_name)
    if not dc_ok:
        return None

    estnu = STNU.from_graphml(out_loc)
    rte_sample = sample_for_rte(real_durations, estnu)
    rte_data = rte_star(estnu, oracle="sample", sample=rte_sample)
    if isinstance(rte_data, bool):
        return None

    start_times, finish_times = get_start_and_finish(estnu, rte_data, len(model.tasks))
    feasible = instance.check_feasibility(
        start_times, finish_times, real_durations, demands
    )
    obj = instance.get_objective(
        [{'task': i, 'start': s, 'end': e} for i, (s, e) in enumerate(zip(start_times, finish_times))],
        objective="makespan"
    )
    sol = {
        'obj': obj if feasible else np.inf,
        'feasibility': feasible,
        'start_times': start_times if feasible else [],
        'time_offline': None,  # Placeholder
        'time_online': None,   # Placeholder
        'method': 'stnu',
        'noise_factor': noise_factor,
        'real_durations': str(real_durations),
    }

    # Plot and save Gantt
    fig, axes = plt.subplots(
        instance.num_resources + 1,
        figsize=(12, 16),
        gridspec_kw={'height_ratios': [6] + [1] * instance.num_resources},
    )
    sol_plot = Solution([TaskData(t.mode, t.resources, s, e)
                         for t, s, e in zip(result_tasks, start_times, finish_times)])
    plot_task_gantt(sol_plot, model.data(), ax=axes[0])
    plot_resource_usage(sol_plot, model.data(), axes=axes[1:])
    plt.savefig(os.path.join(images_folder, f"{base_name}.png"))
    plt.close(fig)

    return sol


def run_stnu_pipeline_fjsp(
    model, instance, duration_distributions, solution, noise_factor, images_folder, instance_folder, sampled_durations):
    stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
    stnu.add_resource_chains(solution.best, model)

    # Export XML and run DC
    timestamp = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")
    base_name = f"{instance_folder}_stnu_{noise_factor}_{timestamp}"
    stnu_to_xml(stnu, base_name, CONFIG['paths']['xml_dir'])
    dc_ok, out_loc = run_dc_algorithm(CONFIG['paths']['xml_dir'], base_name)
    if not dc_ok:
        return None

    estnu = STNU.from_graphml(out_loc)
    rte_sample = sample_for_rte(sampled_durations, estnu)
    rte_data = rte_star(estnu, oracle="sample", sample=rte_sample)
    if isinstance(rte_data, bool):
        return None

    start_times, finish_times = get_start_and_finish(estnu, rte_data, len(model.tasks))
    feasible = instance.check_feasibility(
        start_times, finish_times, sampled_durations)
    simulated_solution, obj = rte_data_to_pyjobshop_solution(solution.best, estnu, rte_data, len(model.tasks), "makespan")

    sol = {
        'obj': obj if feasible else np.inf,
        'feasibility': feasible,
        'start_times': start_times if feasible else [],
        'time_offline': None,  # Placeholder
        'time_online': None,   # Placeholder
        'method': 'stnu',
        'noise_factor': noise_factor,
        'real_durations': str(sampled_durations),
        'time_limit': 20
    }

    # Plot and save Gantt
    fig, axes = plt.subplots(
        instance.num_resources + 1,
        figsize=(12, 16),
        gridspec_kw={'height_ratios': [6] + [1] * instance.num_resources},
    )
    sol_plot = Solution([TaskData(t.mode, t.resources, s, e)
                         for t, s, e in zip(solution.best.tasks, start_times, finish_times)])
    plot_task_gantt(sol_plot, model.data(), ax=axes[0])
    plot_resource_usage(sol_plot, model.data(), axes=axes[1:])
    plt.savefig(os.path.join(images_folder, f"{base_name}.png"))
    plt.close(fig)

    return sol


def process_instances(problem_type, config, logger):
    base_path = config['paths']['base']
    results_stamp = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")
    output_file = f"final_results_{results_stamp}.csv"
    infeasible = { 'stnu': {}, 'proactive': {}, 'reactive': {} }

    for noise in config['experiments']['noise_factors']:
        config['noise_factor'] = noise
        for folder in config['experiments']['instance_folders']:
            folder_path = os.path.join(base_path, 'data', problem_type, folder)
            images_folder = os.path.join(base_path,
                                         config['paths']['image_subdir'],
                                         problem_type,
                                         str(time.time()),
                                         f"noise{noise}",
                                         folder)
            for idx, fname in enumerate(os.listdir(folder_path)):
                if idx >= 100:
                    break
                logger.info(f"Processing {fname} (noise={noise})")
                inst = create_instance(os.path.join(folder_path, fname), problem_type)
                samples, distros = inst.sample_durations(
                    config['experiments']['nb_scenarios_test'], noise
                )
                img_dir = create_output_dirs(folder_path, images_folder, fname)

                for sample in samples:
                    # Proactive & reactive
                    if config['experiments']['proactive_reactive']:
                        run_proactive_and_reactive(
                            inst, sample, config, output_file, folder
                        )
                    # STNU
                    if config['experiments']['stnu']:
                        if (problem_type.startswith("mm")):
                            model = inst.create_model(inst.sample_mode(
                                config['stnu']['mode'], noise
                            ))
                            res = model.solve(time_limit=5, display=False)
                            if not res.best.tasks:
                                logger.info("Infeasible STNU offline solution")
                                continue
                            schedule = inst.get_schedule(res.best.tasks)
                            real_durs = inst.get_real_durations(res.best.tasks, sample)
                            sol = run_stnu_pipeline(
                                model, inst, distros, res.best.tasks,
                                real_durs, schedule, noise, img_dir, folder
                            )
                        else:
                            model = inst.create_model()
                            result = model.solve(time_limit=20, display=False
                                                 #, solver='cpoptimizer'
                                                 )
                            result_tasks = result.best.tasks
                            sampled_durations = inst.get_real_durations(result_tasks, sample)

                            sol = run_stnu_pipeline_fjsp(model, inst, distros, result, noise, img_dir, folder, sampled_durations)
                        if sol:
                            data_to_csv(folder, sol, output_file)
    # Final evaluation
    evaluate_results(now=results_stamp)
    logger.info("Pipeline complete.")


def main():
    args = parse_args()
    logger = setup_logger()
    process_instances(args.problem_type, CONFIG, logger)


if __name__ == "__main__":
    main()