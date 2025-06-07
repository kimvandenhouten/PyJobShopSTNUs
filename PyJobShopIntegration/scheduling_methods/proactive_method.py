import time
import copy
import numpy as np

import general.logger
from rcpsp_max.solvers.check_feasibility import check_feasibility_rcpsp_max

logger = general.logger.get_logger(__name__)


def run_proactive_offline(instance, noise_factor=1, time_limit=60, mode="robust", nb_scenarios_saa=10):
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

    start_offline = time.time()
    # Solve very conservative schedule
    lb, ub = instance.get_bounds(noise_factor=noise_factor)

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
        model = instance.create_model(durations)
        result = model.solve(time_limit=time_limit, display=False, random_seed=42, num_search_workers=1)
        if result:
            start_times = [task.start for task in result.best.tasks]
            estimated_durations = []
            for i, task in enumerate(result.best.tasks):
                mode = task.mode
                estimated_durations.append(durations[mode])
            data_dict["estimated_durations"] = estimated_durations
            data_dict["result_tasks"] = [task for task in result.best.tasks]
    elif mode.startswith("quantile_"):
        quantile = float(mode.split("_")[1])
        if quantile is not None:
            durations = get_quantile(lb, ub, quantile)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        logger.debug(f'Start solving upper bound schedule {durations}')
        model = instance.create_model(durations)
        result = model.solve(time_limit=time_limit, display=False, random_seed=42, num_search_workers=1)
        if result:
            print("Solution found proactively quantile")
            start_times = [task.start for task in result.best.tasks]
            data_dict["result_tasks"] = [task for task in result.best.tasks]
            estimated_durations = []
            for i, task in enumerate(result.best.tasks):
                mode = task.mode
                estimated_durations.append(durations[mode])
            data_dict["estimated_durations"] = estimated_durations

    # elif mode == "SAA":
    #     # Sample scenarios for Sample Average Approximation and solve
    #     train_durations_sample = rcpsp_max.sample_durations(nb_scenarios_saa)
    #     res, start_times = rcpsp_max.solve_saa(train_durations_sample, time_limit)
    #
    # elif mode == "SAA_smart":
    #     # Sample scenarios for Sample Average Approximation and solve
    #     train_durations_sample = []
    #     for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
    #         durations = [int(lb[i] + quantile * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]
    #         train_durations_sample.append(durations)
    #     res, start_times = rcpsp_max.solve_saa(train_durations_sample, time_limit)

    else:
        raise NotImplementedError

    if result:
        logger.debug(f'Robust start times are {start_times}')
        demands = []
        for i, task in enumerate(result.best.tasks):
            mode = task.mode
            demands.append(instance.modes[mode].demands)
        data_dict["demands"] = demands
        data_dict["start_times"] = start_times
        data_dict["time_offline"] = time.time() - start_offline
        data_dict["estimated_start_times"] = start_times

    else:
        logger.debug(f'No robust schedule exists')

    return data_dict


def run_proactive_online(instance, duration_sample, data_dict):
    """
    Evaluate the robust approach
    """
    infeasible = True
    data_dict = copy.deepcopy(data_dict)
    data_dict["real_durations"] = str(duration_sample)
    start_times = data_dict["start_times"]
    if start_times is not None:
        start_online = time.time()
        # Check feasibility
        finish_times = [start_times[i] + duration_sample[i] for i in range(len(start_times))]
        check_feasibility = instance.check_feasibility(start_times, finish_times, duration_sample, data_dict["demands"])
        finish_online = time.time()
        if check_feasibility:
            logger.info(f'Proactive schedule is feasible with sample {duration_sample}')
            data_dict["feasibility"] = True
            data_dict["time_online"] = finish_online - start_online
            schedule = instance.get_schedule(data_dict["result_tasks"])
            data_dict["obj"] = instance.get_objective(schedule)
            infeasible = False

    if infeasible:
        logger.info(f'Proactive schedule is infeasible with sample {duration_sample}')
    return data_dict
