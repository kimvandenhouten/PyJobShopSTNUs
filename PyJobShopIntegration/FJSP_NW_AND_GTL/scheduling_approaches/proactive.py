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

    # ------------------------------------------------------------------
    # FAST *no-wait* REPAIR  –  push each successor op forward so that
    # start(successor) = finish(predecessor) whenever those two belong
    # to the same job and are linked by an end-before-start (delay 0).
    # ------------------------------------------------------------------
    changed = True
    while changed:  # iterate until no further shifts
        changed = False
        for cons in model.constraints.end_before_start:
            if cons.delay != 0:  # only strict no-wait arcs
                continue
            i, j = cons.task1, cons.task2
            fi = finish_times[i]
            sj = start_times[j]
            if sj < fi:  # successor starts too early
                delta = fi - sj
                start_times[j] += delta
                finish_times[j] += delta
                changed = True

    feasible = check_feasibility(
        model, start_times, finish_times, result.best.tasks
    )
    t1 = time.time()

    if feasible:
        data["time_online"] = t1 - t0
        data["obj"]         = max(finish_times)
        data["feasibility"] = True
        data["start_times"] = start_times
        data["finish_times"] = finish_times
    return data

# def run_proactive_online_direct(duration_sample, data_dict, result, fjsp_instance):
#     t0 = time.time()
#     data = copy.deepcopy(data_dict)
#     data["real_durations"] = [int(x) for x in duration_sample]
#
#     model = fjsp_instance.model
#     task_list = result.best.tasks  # baseline order
#     n_tasks = len(model.tasks)
#
#     # ------------------------------------------------------------------
#     # 1. helpers: predecessor map & per-machine baseline sequence
#     # ------------------------------------------------------------------
#     pred_map = defaultdict(list)  # task → list[(pred, delay, 'kind')]
#     for e in model.constraints.end_before_start:
#         pred_map[e.task2].append((e.task1, e.delay, 'EBS'))
#     for s in model.constraints.start_before_start:
#         pred_map[s.task2].append((s.task1, s.delay, 'SBS'))
#     for s in model.constraints.start_before_end:
#         pred_map[s.task2].append((s.task1, s.delay, 'SBE'))
#     for e in model.constraints.end_before_end:
#         pred_map[e.task2].append((e.task1, e.delay, 'EBE'))
#
#     seq_on_machine = defaultdict(list)  # machine → ordered [(task, mode)]
#     for td in task_list:
#         m = td.resources[0]
#         seq_on_machine[m].append((model.modes[td.mode].task, td.mode))
#
#     setup = {(st.machine, st.task1, st.task2): st.duration
#              for st in getattr(model.constraints, "setup_times", ())}
#
#     # ------------------------------------------------------------------
#     # 2. build initial start / finish vectors
#     # ------------------------------------------------------------------
#     start = np.array(data_dict["start_times"], dtype=int)
#     finish = start.copy()
#     # duration_sample is per-mode; map it to the mode actually chosen
#     for td in task_list:
#         mode_idx = td.mode
#         task_idx = model.modes[mode_idx].task
#         real_dur = int(duration_sample[mode_idx])
#         finish[task_idx] = start[task_idx] + real_dur
#
#     # ------------------------------------------------------------------
#     # 3-A. machine-capacity + SDST repair
#     # ------------------------------------------------------------------
#     for m, chain in seq_on_machine.items():
#         prev_task, prev_fin = None, 0
#         for tk, md in chain:
#             s = max(start[tk], prev_fin + setup.get((m, prev_task, tk), 0))
#             if s > start[tk]:
#                 delta = s - start[tk]
#                 start[tk] += delta
#                 finish[tk] += delta
#             prev_task, prev_fin = tk, finish[tk]
#
#     # ------------------------------------------------------------------
#     # 3-B. precedence / no-wait propagation  (bounded iterations)
#     # ------------------------------------------------------------------
#     for _ in range(n_tasks):  # upper bound in DAG
#         changed = False
#         for succ in range(n_tasks):
#             for pred, dly, kind in pred_map[succ]:
#                 if kind == 'EBS':  # finish(pred)+d ≤ start(succ)
#                     needed = finish[pred] + dly
#                     stamp = start[succ]
#                 elif kind == 'SBS':  # start(pred)+d ≤ start(succ)
#                     needed = start[pred] + dly
#                     stamp = start[succ]
#                 elif kind == 'SBE':  # start(pred)+d ≤ finish(succ)
#                     needed = start[pred] + dly
#                     stamp = finish[succ]
#                 else:  # 'EBE'           # finish(pred)+d ≤ finish(succ)
#                     needed = finish[pred] + dly
#                     stamp = finish[succ]
#
#                 if stamp < needed:
#                     delta = needed - stamp
#                     start[succ] += delta
#                     finish[succ] += delta
#                     changed = True
#         if not changed:
#             break
#     else:
#         # exceeded max iterations → probably a cycle; mark infeasible
#         logger.warning("Precedence propagation failed to converge.")
#         data.update(time_online=time.time() - t0,
#                     feasibility=False,
#                     obj=np.inf)
#         return data
#
#     # ------------------------------------------------------------------
#     # 4. final feasibility check  (skip duration-equality)
#     # ------------------------------------------------------------------
#     feasible = check_feasibility(model,
#                                  [int(x) for x in start],
#                                  [int(x) for x in finish],
#                                  task_list)
#
#     # ------------------------------------------------------------------
#     # 5. fill record and return
#     # ------------------------------------------------------------------
#     data["time_online"] = time.time() - t0
#     data["feasibility"] = bool(feasible)
#
#     if feasible:
#         data["start_times"] = [int(x) for x in start]
#         data["finish_times"] = [int(x) for x in finish]
#         data["obj"] = int(finish.max())
#     else:
#         data["obj"] = np.inf
#
#     return data