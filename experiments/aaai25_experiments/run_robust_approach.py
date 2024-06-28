# Import
import numpy as np
import time
import general
from general.logger import get_logger
from rcpsp_max.solvers.check_feasibility import check_feasibility_rcpsp_max
import copy

logger = general.logger.get_logger(__name__)


def run_robust_offline(rcpsp_max, time_limit=60, mode="robust"):
    # Initialize data
    data_dict = {
        "instance_folder": rcpsp_max.instance_folder,
        "instance_id": rcpsp_max.instance_id,
        "method": f"robust_{mode}",
        "time_limit": time_limit,
        "feasibility": False,
        "obj": np.inf,
        "time_offline": np.inf,
        "time_online": np.inf,
        "start_times": None,
        "real_durations": None
    }

    start_offline = time.time()
    # Solve very conservative schedule
    if mode == "robust":
        durations = rcpsp_max.get_bound()
        logger.debug(f'Start solving upper bound schedule {durations}')
    elif mode == "quantile_0.9":
        lb = rcpsp_max.get_bound(mode="lower_bound")
        ub = rcpsp_max.get_bound(mode="upper_bound")
        durations = [int(lb[i] + 0.9 * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]

        res, data = rcpsp_max.solve(durations, time_limit=time_limit, mode="Quiet")
    else:
        raise NotImplementedError

    if res:
        start_times = data['start'].tolist()
        logger.debug(f'Robust start times are {start_times}')
        data_dict["start_times"] = start_times
        data_dict["time_offline"] = time.time() - start_offline

    else:
        logger.debug(f'No robust schedule exists')

    return data_dict


def run_robust_online(rcpsp_max, duration_sample, data_dict):
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









