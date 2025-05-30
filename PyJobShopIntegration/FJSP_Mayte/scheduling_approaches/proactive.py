import time
import copy
from collections import defaultdict

import numpy as np
import general.logger
from utils import check_feasibility_fjsp

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


def run_proactive_online_cp(duration_sample, data_dict, result, fjsp_instance):
    data = copy.deepcopy(data_dict)
    data["real_durations"] = str(duration_sample)

    # 3) Rebuild a Model with these new durations
    new_model = fjsp_instance.model_new_durations(duration_sample)

    start_online = time.time()
    result_online = new_model.solve(
        display=False,
        initial_solution=result.best # warm start solver with previous solution
    )
    finish_online = time.time()

    if result_online.status in {"Feasible", "Optimal"}:
        data["feasibility"] = True
        data["time_online"] = finish_online - start_online
        data["obj"] = result_online.objective

    return data

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
    start_times, finish_times = compute_finish_times(
        duration_sample=duration_sample,
        task_data_list=result.best.tasks,
        modes=model.modes,
        n_tasks=n_tasks,
        setup_times=setup_times
    )

    feasible = check_feasibility(
        model, start_times, finish_times, result.best.tasks
    )
    t1 = time.time()

    # 4. record timing & makespan
    data["time_online"] = t1 - t0
    data["obj"]         = max(finish_times)
    data["feasibility"] = feasible

    return data

def compute_finish_times(duration_sample: dict[int,int],
                         task_data_list: list,
                         modes: list,
                         n_tasks: int,
                         setup_times: dict[tuple[int,int,int], int]):

    # group by machine/resource
    by_res = defaultdict(list)
    for td in task_data_list:
        res = td.resources[0]
        by_res[res].append(td)

    # prepare output arrays
    start_times  = [0] * n_tasks
    finish_times = [0] * n_tasks
    get_setup = setup_times.get

    # inject setups and recompute in each resource’s order
    for res, ops in by_res.items():
        ops.sort(key=lambda td: td.start)
        prev_task = None
        prev_end  = None

        for td in ops:
            task_idx = modes[td.mode].task
            # lookup any sequence‐dependent setup
            setup = get_setup((res, prev_task, task_idx), 0) if prev_task is not None else 0

            # actual start = max(nominal CP start, prev_end + setup)
            if prev_end is None:
                actual_start = td.start
            else:
                earliest = prev_end + setup
                actual_start = td.start if td.start > earliest else earliest

            actual_finish = actual_start + duration_sample[task_idx]

            start_times[task_idx]  = actual_start
            finish_times[task_idx] = actual_finish

            prev_task = task_idx
            prev_end  = actual_finish

    return start_times, finish_times

def check_feasibility(model, start_times, finish_times, task_data_list):
    """
    Returns True iff no temporal constraint in `model` is violated
    by the (start_times, finish_times) for the given solution.
    """

    # 1) duration consistency
    # for td in task_data_list:
    #     task_idx = model.modes[td.mode].task
    #     dur      = finish_times[task_idx] - start_times[task_idx]
    #     if dur != model.modes[td.mode].duration:
    #         logger.debug(f"Duration mismatch on task {task_idx}: "
    #                       f"{dur} vs {model.modes[td.mode].duration}")
    #         return False

    # 2) start_before_start: start[t1] + delay <= start[t2]
    for cons in model.constraints.start_before_start:
        i, j, d = cons.task1, cons.task2, cons.delay
        if start_times[i] + d > start_times[j]:
            logger.debug(f"SBS violation: start[{i}]+{d}={start_times[i]+d} > start[{j}]={start_times[j]}")
            return False

    # 3) start_before_end: start[t1] + delay <= finish[t2]
    for cons in model.constraints.start_before_end:
        i, j, d = cons.task1, cons.task2, cons.delay
        if start_times[i] + d > finish_times[j]:
            logger.debug(f"SBE violation: start[{i}]+{d}={start_times[i]+d} > finish[{j}]={finish_times[j]}")
            return False

    # 4) end_before_start: finish[t1] + delay <= start[t2]
    for cons in model.constraints.end_before_start:
        i, j, d = cons.task1, cons.task2, cons.delay
        if finish_times[i] + d > start_times[j]:
            logger.debug(f"EBS violation: finish[{i}]+{d}={finish_times[i]+d} > start[{j}]={start_times[j]}")
            return False

    # 5) end_before_end: finish[t1] + delay <= finish[t2]
    for cons in model.constraints.end_before_end:
        i, j, d = cons.task1, cons.task2, cons.delay
        if finish_times[i] + d > finish_times[j]:
            logger.debug(f"EBE violation: finish[{i}]+{d}={finish_times[i]+d} > finish[{j}]={finish_times[j]}")
            return False

    # 6) identical_resources: ensure two tasks use the *same* resource (you can skip if not used)
    for cons in model.constraints.identical_resources:
        i, j = cons.task1, cons.task2
        res_i = model.modes[task_data_list[i].mode].resources
        res_j = model.modes[task_data_list[j].mode].resources
        if set(res_i) != set(res_j):
            logger.debug(f"Identical-resources violation on {i},{j}: {res_i} vs {res_j}")
            return False

    # 7) different_resources: ensure two tasks use *different* resources
    for cons in model.constraints.different_resources:
        i, j = cons.task1, cons.task2
        res_i = set(model.modes[task_data_list[i].mode].resources)
        res_j = set(model.modes[task_data_list[j].mode].resources)
        if res_i & res_j:
            logger.debug(f"Different-resources violation on {i},{j}: intersection {res_i & res_j}")
            return False

    # 8) setup-time & machine‐capacity (SDST)
    #    Rebuild each machine’s run‐sequence, sort by start, then check
    #    start[j] ≥ finish[i] + setup_time(m,i,j)
    setup_lookup = {
        (st.machine, st.task1, st.task2): st.duration
        for st in model.constraints.setup_times
    }
    by_machine = defaultdict(list)
    for td in task_data_list:
        m = td.resources[0]
        t = model.modes[td.mode].task
        by_machine[m].append(t)

    for m, tasks in by_machine.items():
        tasks.sort(key=lambda t: start_times[t])
        for i, j in zip(tasks, tasks[1:]):
            sdu = setup_lookup.get((m, i, j), 0)
            if start_times[j] < finish_times[i] + sdu:
                logger.debug(
                    f"SDST violation on machine {m}: "
                    f"finish[{i}]={finish_times[i]} + setup={sdu} > start[{j}]={start_times[j]}"
                )
                return False

    return True