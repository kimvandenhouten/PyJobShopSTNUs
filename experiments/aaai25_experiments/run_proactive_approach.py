import time
import copy
import numpy as np

import general.logger
from rcpsp_max.solvers.check_feasibility import check_feasibility_rcpsp_max

logger = general.logger.get_logger(__name__)


def run_proactive_offline(rcpsp_max, time_limit=60, mode="robust", nb_scenarios_saa=10):
    # Initialize data
    data_dict = {
        "instance_folder": rcpsp_max.instance_folder,
        "instance_id": rcpsp_max.instance_id,
        "noise_factor": rcpsp_max.noise_factor,
        "method": f"proactive_{mode}",
        "time_limit": time_limit,
        "feasibility": False,
        "obj": np.inf,
        "time_offline": np.inf,
        "time_online": np.inf,
        "start_times": None,
        "real_durations": None,
        "mode": mode
    }

    start_offline = time.time()
    # Solve very conservative schedule
    lb = rcpsp_max.get_bound(mode="lower_bound")
    ub = rcpsp_max.get_bound(mode="upper_bound")

    def get_quantile(lb, ub, p):
        if lb == ub:
            quantile = lb
        quantile = [int(lb[i] + p * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]

        return quantile

    if mode == "robust":
        durations = ub
        logger.debug(f'Start solving upper bound schedule {durations}')
        data_dict["estimated_durations"] = durations
        res, data = rcpsp_max.solve(durations, time_limit=time_limit, mode="Quiet")
        if res:
            start_times = data['start'].tolist()

    elif mode == "quantile_0.25":
        durations = get_quantile(lb, ub, 0.25)
        data_dict["estimated_durations"] = durations
        logger.debug(f'Start solving upper bound schedule {durations}')
        res, data = rcpsp_max.solve(durations, time_limit=time_limit, mode="Quiet")
        if res:
            start_times = data['start'].tolist()

    elif mode == "quantile_0.75":
        durations = get_quantile(lb, ub, 0.75)
        data_dict["estimated_durations"] = durations
        logger.debug(f'Start solving upper bound schedule {durations}')
        res, data = rcpsp_max.solve(durations, time_limit=time_limit, mode="Quiet")
        if res:
            start_times = data['start'].tolist()

    elif mode == "quantile_0.5":
        durations = get_quantile(lb, ub, 0.5)
        data_dict["estimated_durations"] = durations
        logger.debug(f'Start solving upper bound schedule {durations}')
        res, data = rcpsp_max.solve(durations, time_limit=time_limit, mode="Quiet")
        if res:
            start_times = data['start'].tolist()

    elif mode == "quantile_0.9":
        durations = get_quantile(lb, ub, 0.9)
        data_dict["estimated_durations"] = durations
        logger.debug(f'Start solving upper bound schedule {durations}')
        res, data = rcpsp_max.solve(durations, time_limit=time_limit, mode="Quiet")
        if res:
            start_times = data['start'].tolist()

    elif mode == "SAA":
        # Sample scenarios for Sample Average Approximation and solve
        train_durations_sample = rcpsp_max.sample_durations(nb_scenarios_saa)
        res, start_times = rcpsp_max.solve_saa(train_durations_sample, time_limit)

    elif mode == "SAA_smart":
        # Sample scenarios for Sample Average Approximation and solve
        train_durations_sample = []
        for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
            durations = [int(lb[i] + quantile * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]
            train_durations_sample.append(durations)
        res, start_times = rcpsp_max.solve_saa(train_durations_sample, time_limit)

    else:
        raise NotImplementedError

    if res:
        logger.debug(f'Robust start times are {start_times}')
        data_dict["start_times"] = start_times
        data_dict["time_offline"] = time.time() - start_offline
        data_dict["estimated_start_times"] = start_times

    else:
        logger.debug(f'No robust schedule exists')

    return data_dict


def run_proactive_online(rcpsp_max, duration_sample, data_dict):
    """
    Evaluate the robust approach
    """
    infeasible = True
    data_dict = copy.deepcopy(data_dict)
    data_dict["real_durations"] = duration_sample
    start_times = data_dict["start_times"]
    if start_times is not None:
        start_online = time.time()
        # Check feasibility
        finish_times = [start_times[i] + duration_sample[i] for i in range(len(start_times))]
        check_feasibility = check_feasibility_rcpsp_max(start_times, finish_times, duration_sample, rcpsp_max.capacity,
                                    rcpsp_max.needs, rcpsp_max.temporal_constraints)
        finish_online = time.time()
        if check_feasibility:
            logger.info(f'{rcpsp_max.instance_folder}_PSP{rcpsp_max.instance_id} FEASIBLE with makespan {max(finish_times)} with sample {duration_sample}')
            data_dict["feasibility"] = True
            data_dict["time_online"] = finish_online - start_online
            data_dict["obj"] = max(finish_times)
            infeasible = False

    if infeasible:
        logger.info(
            f'{rcpsp_max.instance_folder}_PSP{rcpsp_max.instance_id} INFEASIBLE with sample {duration_sample}')

    return [data_dict]




