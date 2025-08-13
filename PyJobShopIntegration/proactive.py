import time
import copy
from collections import defaultdict

import numpy as np
import general.logger
from FJSP import compute_finish_times, check_feasibility

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

def check_feasibility2(model, finish_times, start_times, solution, setup_times):
    data = model.data()  # returns ProblemData

    # # 1) Build mode_id → (task_id, machine_id) mapping
    mode_to_task_machine = {
        mode_index: (mode.task, mode.resources[0])
        for mode_index, mode in enumerate(data.modes)
    }

    # 2) Precedence constraints with setup
    for c in getattr(data.constraints, "end_before_start", []):
        t1, t2, delay = c.task1, c.task2, c.delay
        finish_t1 = finish_times[t1]
        start_t2 = start_times[t2]

        # Check if both use same machine
        _, m1 = mode_to_task_machine[solution.tasks[t1].mode]
        _, m2 = mode_to_task_machine[solution.tasks[t2].mode]
        setup = setup_times.get((m1, t1, t2), 0) if m1 == m2 else 0

        if finish_t1 + delay + setup > start_t2:
            return False

    # 3) SDST constraints per machine
    tasks_on_machine = {}
    for td in solution.tasks:
        task_id, m = mode_to_task_machine[td.mode]
        tasks_on_machine.setdefault(m, []).append((td.start, td.end, task_id))

    for m, intervals in tasks_on_machine.items():
        intervals.sort()
        for (s1, e1, t1), (s2, e2, t2) in zip(intervals, intervals[1:]):
            setup = setup_times.get((m, t1, t2), 0)
            if e1 + setup > s2:
                return False

    return True

def run_proactive_online_direct(duration_sample, data_dict, result, fjsp_instance):
    data = copy.deepcopy(data_dict)
    data["real_durations"] = str(duration_sample)

    model = fjsp_instance.model

    # 2. build setup lookup
    setup_times = {
        (st.machine, st.task1, st.task2): st.duration
        for st in getattr(model.constraints, "setup_times", ())
    }
    start_times = data["start_times"]
    finish_times = [0] * len(start_times)

    # 3. recompute start/finish vectors
    t0 = time.time()

    for task_data in result.best.tasks:
        mode_id = task_data.mode
        task_id = model.modes[mode_id].task
        duration = duration_sample[mode_id]
        finish_times[task_id] = start_times[task_id] + duration

    feasible = check_feasibility2(model, finish_times, start_times,result.best, setup_times)
    t1 = time.time()

    if feasible:
        data["time_online"] = t1 - t0
        data["obj"]         = max(finish_times)
        data["feasibility"] = True
    return data
