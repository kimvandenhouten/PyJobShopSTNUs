import time
import copy
from collections import defaultdict

import numpy as np
import general.logger
from PyJobShopIntegration.FJSP_NW_AND_GTL.FJSP import (compute_finish_times, check_feasibility)

logger = general.logger.get_logger(__name__)


def run_proactive_offline(fjsp_instance, noise_factor, time_limit, mode):
    # Initialize data
    data_dict = {'obj': np.inf,
                 'feasibility': False,
                 'start_times': None,
                 'time_online': np.inf,
                 'time_offline': np.inf,
                 'noise_factor': noise_factor,
                 'method': f'proactive_{mode}',
                 'time_limit': time_limit,
                 'real_durations': None,
                 'estimated_durations': None,
                 'result_tasks:': None,
                 }

    lb, ub = fjsp_instance.get_bounds(noise_factor=noise_factor)

    def get_quantile(lb, ub, p):
        if lb == ub:
            quantile = lb
        else:
            quantile = [int(lb[k] + p * (ub[k] - lb[k] + 1) - 1) for k in range(len(lb))]

        return quantile

    quantile_map = {
        "quantile_0.25": 0.25,
        "quantile_0.5": 0.5,
        "quantile_0.75": 0.75,
        "quantile_0.9": 0.9,
    }

    if mode == "robust":
        durations = ub
        logger.debug(f'Start solving upper bound schedule {durations}')
        model = fjsp_instance.model_new_durations(durations)
        start_offline = time.time()
        result = model.solve(solver='cpoptimizer', time_limit=time_limit, display=False)
        update_dict(data_dict=data_dict, durations=durations, result=result, time_offline= time.time() - start_offline)

    elif mode.startswith("quantile_"):
        quantile = float(mode.split("_")[1])
        if quantile is not None:
            durations = get_quantile(lb, ub, quantile)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        logger.debug(f'Start solving upper bound schedule {durations}')
        model = fjsp_instance.model_new_durations(durations)
        start_offline = time.time()
        result = model.solve(solver='cpoptimizer', time_limit=time_limit, display=False)
        update_dict(data_dict=data_dict, durations=durations, result=result, time_offline=time.time() - start_offline)

    else:
        logger.debug(f'No robust schedule exists')

    return data_dict, result

def update_dict(data_dict, durations, result, time_offline):
    if result:
        start_times = [task.start for task in result.best.tasks]
        logger.debug(f'Robust start times are {start_times}')
        data_dict["time_offline"] = time_offline
        estimated_durations = []
        for i, task in enumerate(result.best.tasks):
            mode = task.mode
            estimated_durations.append(durations[mode])
        data_dict["estimated_durations"] = estimated_durations
        data_dict["result_tasks"] = [task for task in result.best.tasks]
        data_dict["start_times"] = start_times


# def run_proactive_online_cp(duration_sample, data_dict, result, fjsp_instance):
#     data = copy.deepcopy(data_dict)
#     data["real_durations"] = str(duration_sample)
#
#     # 3) Rebuild a Model with these new durations
#     new_model = fjsp_instance.model_new_durations(duration_sample)
#
#     start_online = time.time()
#     result_online = new_model.solve(
#         display=False,
#         initial_solution=result.best # warm start solver with previous solution
#     )
#     finish_online = time.time()
#
#     if result_online.status in {"Feasible", "Optimal"}:
#         data["feasibility"] = True
#         data["time_online"] = finish_online - start_online
#         data["obj"] = result_online.objective
#
#     return data

def run_proactive_online_direct(duration_sample, data_dict, result, fjsp_instance):
    data = copy.deepcopy(data_dict)
    data["real_durations"] = str(duration_sample)

    model   = fjsp_instance.model
    n_tasks = len(model.tasks)

    # 2. build setup lookup
    setup_times = {
        (st.machine, st.task1, st.task2): st.duration
        for st in getattr(model.constraints, "setup_times", ())
    }

    # 3. recompute start/finish vectors
    t0 = time.time()
    if (model.constraints.setup_times):
        start_times, finish_times = compute_finish_times(
            duration_sample=duration_sample,
            task_data_list=result.best.tasks,
            modes=model.modes,
            n_tasks=n_tasks,
            setup_times=setup_times
        )
    else:
        start_times = data_dict["start_times"]
        finish_times = [start_times[i] + duration_sample[i] for i in range(len(start_times))]

    feasible = check_feasibility(
        model, start_times, finish_times, result.best.tasks
    )
    t1 = time.time()

    if feasible:
        data["time_online"] = t1 - t0
        data["obj"]         = max(finish_times)
        data["feasibility"] = True
    return data