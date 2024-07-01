import time
import copy
import numpy as np

import general
from rcpsp_max.solvers.check_feasibility import check_feasibility_rcpsp_max
from rcpsp_max.temporal_networks.stnu_rcpsp_max import RCPSP_STNU, get_resource_chains, add_resource_chains
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU

logger = general.logger.get_logger(__name__)


def get_start_and_finish(estnu, rte_data, num_tasks):
    """
    This function can be used to link the start times and finish times from the rte_dta
    to the RCPSP_max starts and finish times
    """
    true_durations, start_times, finish_times = [], [], []
    for task in range(num_tasks):
        if task > 0 and task < num_tasks - 1:
            node_idx_start = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_START}']
            node_idx_finish = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_FINISH}']
            start_times.append(rte_data.f[node_idx_start])
            finish_times.append(rte_data.f[node_idx_finish])
        else:
            node_idx_start = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_START}']
            node_idx_finish = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_START}']
            start_times.append(rte_data.f[node_idx_start])
            finish_times.append(rte_data.f[node_idx_finish])
    return start_times, finish_times


def get_true_durations(estnu, rte_data, num_tasks):
    """
    This function can be used to return the durations according to the RTE schedule
    related to the RCPSP_max
    """
    start_times, finish_times = get_start_and_finish(estnu, rte_data, num_tasks)
    true_durations = [finish_times[i] - start_times[i] for i in range(num_tasks)]

    return true_durations


def run_stnu_offline(rcpsp_max, time_limit_cp_stnu=60, mode="mean"):

    data_dict = {
        "instance_folder": rcpsp_max.instance_folder,
        "instance_id": rcpsp_max.instance_id,
        "noise_factor": rcpsp_max.noise_factor,
        "method": f"STNU_{mode}",
        "time_limit_cp_stnu": time_limit_cp_stnu,
        "can_build_stnu": None,
        "dc": None,
        "obj": None,
        "feasibility": None,
        "real_durations": None,
        "start_times": None,
        "time_offline": np.inf,
        "time_online": np.inf,
        "mode": mode
    }

    logger.debug(f'Start instance {rcpsp_max.instance_id}')
    start_offline = time.time()

    # Read instance and set up deterministic RCPSP/max CP model and solve (this will be used for the resource chain)
    if mode == "mean":
        lb = rcpsp_max.get_bound(mode="lower_bound")
        ub = rcpsp_max.get_bound(mode="upper_bound")
        durations = [int((lb[i] + ub[i]) / 2) for i in range(rcpsp_max.num_tasks)]
    elif mode == "quantile_0.25":
        lb = rcpsp_max.get_bound(mode="lower_bound")
        ub = rcpsp_max.get_bound(mode="upper_bound")
        durations = [int(lb[i] + 0.25 * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]
    elif mode == "quantile_0.75":
        lb = rcpsp_max.get_bound(mode="lower_bound")
        ub = rcpsp_max.get_bound(mode="upper_bound")
        durations = [int(lb[i] + 0.75 * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]
    elif mode == "quantile_0.9":
        lb = rcpsp_max.get_bound(mode="lower_bound")
        ub = rcpsp_max.get_bound(mode="upper_bound")
        durations = [int(lb[i] + 0.9 * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]
    elif mode == "robust":
        durations = rcpsp_max.get_bound(mode="upper_bound")
        logger.info(f'Upper bound used for making the STNU is {durations}')
    else:
        raise NotImplementedError(f'Mode {mode} not recognized')

    res, schedule = rcpsp_max.solve(durations, time_limit=time_limit_cp_stnu)

    if res:
        data_dict['can_build_stnu'] = True
        # Build the STNU using the instance information and the resource chains
        schedule = schedule.to_dict('records')
        resource_chains, resource_assignments = get_resource_chains(schedule, rcpsp_max.capacity, rcpsp_max.needs,
                                                                    complete=True)
        stnu = RCPSP_STNU.from_rcpsp_max_instance(rcpsp_max.durations, rcpsp_max.temporal_constraints, noise_factor=rcpsp_max.noise_factor)
        stnu = add_resource_chains(stnu, resource_chains)
        stnu_to_xml(stnu, f"example_rcpsp_max_stnu", "temporal_networks/cstnu_tool/xml_files")

        # Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
        dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files",
                                               f"example_rcpsp_max_stnu")
        logger.debug(f'dc is {dc} and output location {output_location}')

        # Read ESTNU xml file into Python object that was the output from the previous step
        estnu = STNU.from_graphml(output_location)
        finish_offline = time.time()
        data_dict['time_offline'] = finish_offline - start_offline
    else:
        estnu = None
        dc = False

    data_dict['dc'] = dc
    return dc, estnu, data_dict


def run_stnu_online(dc, estnu, sample_duration, rcpsp_max, data_dict):
    data_dict = copy.deepcopy(data_dict)
    data_dict['real_durations'] = sample_duration
    data_dict['obj'] = np.inf
    data_dict['feasibility'] = False
    data_dict['time_online'] = np.inf
    feasibility = False
    logger.info(f'{rcpsp_max.instance_folder}_PSP{rcpsp_max.instance_id} STNU {data_dict["mode"]} is dc {dc} ')
    if estnu is not None:
        if dc:
            start_online = time.time()
            # Transform the sample_duration to a dictionary and find the correct ESTNU node indices
            sample = {}
            for task, duration in enumerate(sample_duration):
                if 0 >= task or task >= len(sample_duration) - 1:  # skip the source and sink node
                    continue
                find_contingent_node = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_FINISH}']
                sample[find_contingent_node] = duration

            logger.debug(f'Sample dict that will be given to RTE star is {sample_duration}')

            # Run RTE algorithm with alternative oracle and store makespan
            rte_data = rte_star(estnu, oracle="sample", sample=sample)

            if rte_data:
                objective = max(rte_data.f.values())

                # Obtain true durations and start and finish times and verify feasibility
                start_times, finish_times = get_start_and_finish(estnu, rte_data, rcpsp_max.num_tasks)

                feasibility = check_feasibility_rcpsp_max(start_times, finish_times, sample_duration, rcpsp_max.capacity, rcpsp_max.needs,
                                           rcpsp_max.temporal_constraints)
                finish_online = time.time()
                data_dict['obj'] = objective
                data_dict['feasibility'] = feasibility
                data_dict['start_times'] = start_times
                data_dict['time_online'] = finish_online - start_online
                logger.info(
                    f'{rcpsp_max.instance_folder}_PSP{rcpsp_max.instance_id} is FEASIBLE with makespan {objective} with true durations {sample_duration}')
            else:
                raise ValueError(f'WARNING: for some reason the RTE could not finish')
                data_dict['warning'] = "RTE_error"
    if not feasibility:
        logger.info(
            f'{rcpsp_max.instance_folder}_PSP{rcpsp_max.instance_id} is INFEASIBLE with true durations {sample_duration}')

    return [data_dict]

