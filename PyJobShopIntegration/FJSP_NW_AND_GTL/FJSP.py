from collections import defaultdict

import numpy as np
from pyjobshop import Mode, Model

from PyJobShopIntegration.Sampler import DiscreteUniformSampler
import general.logger
logger = general.logger.get_logger(__name__)

class FJSP():

    def __init__(self, model):
        self.model = model

    def get_durations(self):
        return [mode.duration for mode in self.model.modes]

    def model_new_durations(self, new_durations):

        data = self.model.data()

        # 2) Sanity‐check
        if len(new_durations) != len(data.modes):
            raise ValueError(
                f"Length mismatch: model has {len(data.modes)} modes, "
                f"but you passed {len(new_durations)} durations."
            )

        # 3) Build a new list of Mode objects, copying everything except duration
        new_modes = [
            Mode(
                task=mode.task,
                resources=mode.resources,
                duration=new_d,
                demands=mode.demands
            )
            for mode, new_d in zip(data.modes, new_durations)
        ]

        new_data = data.replace(modes=new_modes)
        return Model.from_data(new_data)

    def duration_distributions(self, noise_factor):
        lower_bound, upper_bound = self.get_bounds(noise_factor)
        duration_distributions = DiscreteUniformSampler(
            lower_bounds=lower_bound,
            upper_bounds=upper_bound
        )
        return duration_distributions

    def get_bounds(self, noise_factor):
        lb = []
        ub = []
        for duration in self.get_durations():
            if duration == 0:
                lb.append(0)
                ub.append(0)
            else:
                lower_bound = int(max(1, duration - noise_factor * np.sqrt(duration)))
                upper_bound = int(duration + noise_factor * np.sqrt(duration))
                if lower_bound == upper_bound:
                    upper_bound += 1
                lb.append(lower_bound)
                ub.append(upper_bound)
        return lb, ub

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

    #1) duration consistency (if no setup_times)
    if not model.constraints.setup_times:
        for td in task_data_list:
            task_idx = model.modes[td.mode].task
            dur      = finish_times[task_idx] - start_times[task_idx]
            if dur != model.modes[td.mode].duration:
                logger.debug(f"Duration mismatch on task {task_idx}: "
                              f"{dur} vs {model.modes[td.mode].duration}")
                return False

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