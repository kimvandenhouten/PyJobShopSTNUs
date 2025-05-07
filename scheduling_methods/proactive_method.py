import time
import copy
import numpy as np
import general.logger
from rcpsp_max.solvers.check_feasibility import check_feasibility_rcpsp_max

logger = general.logger.get_logger(__name__)

def run_proactive_offline(rcpsp_max, time_limit=60, mode="robust", nb_scenarios_saa=10):
    """
    Build a ‘robust’/quantile schedule for RCPSP_MAX.
    Returns a data_dict with:
      - estimated_durations
      - start_times
      - time_offline
      - (other metadata)
    """
    data_dict = {
        "instance_folder": rcpsp_max.instance_folder,
        "instance_id": rcpsp_max.instance_id,
        "noise_factor": rcpsp_max.noise_factor,
        "method": f"proactive_{mode}",
        "time_limit": time_limit,
        "feasibility": False,
        "obj": np.inf,
        "time_offline": None,
        "time_online": None,
        "start_times": None,
        "real_durations": None,
        "mode": mode
    }

    start_offline = time.time()
    lb = rcpsp_max.get_bound(mode="lower_bound")
    ub = rcpsp_max.get_bound(mode="upper_bound")

    def get_quantile(lb, ub, p):
        if lb == ub:
            return lb
        return [int(lb[i] + p * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]

    # initialize
    start_times = None
    durations = None

    if mode == "robust":
        durations = ub

    elif mode.startswith("quantile"):
        p = float(mode.split("_")[1])
        durations = get_quantile(lb, ub, p)

    elif mode == "SAA":
        train_samples = rcpsp_max.sample_durations(nb_scenarios_saa)
        res, start_times = rcpsp_max.solve_saa(train_samples, time_limit)

    elif mode == "SAA_smart":
        train_samples = []
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            train_samples.append(get_quantile(lb, ub, q))
        res, start_times = rcpsp_max.solve_saa(train_samples, time_limit)

    else:
        raise NotImplementedError(f"Unknown mode {mode}")

    # solve CP for robust/quantile
    if mode in ["robust"] or mode.startswith("quantile"):
        data_dict["estimated_durations"] = durations
        logger.debug(f"Solving schedule with durations {durations}")
        res, data = rcpsp_max.solve(durations, time_limit=time_limit, mode="Quiet")
        if res:
            start_times = data["start"].tolist()

    data_dict["time_offline"] = time.time() - start_offline
    data_dict["estimated_durations"] = durations
    data_dict["start_times"] = start_times

    return data_dict


def run_proactive_online(rcpsp_max, duration_sample, data_dict):
    """
    Given an offline data_dict, evaluate online feasibility for one sample.
    Returns a new data_dict (not wrapped in a list).
    """
    result = copy.deepcopy(data_dict)
    result["real_durations"] = duration_sample

    start_times = result.get("start_times")
    if start_times is not None:
        start_online = time.time()
        finish_times = [start_times[i] + duration_sample[i] for i in range(len(start_times))]

        feasible = check_feasibility_rcpsp_max(
            start_times,
            finish_times,
            duration_sample,
            rcpsp_max.capacity,
            rcpsp_max.needs,
            rcpsp_max.temporal_constraints,
        )

        result["time_online"] = time.time() - start_online
        result["feasibility"] = feasible
        result["obj"] = max(finish_times) if feasible else np.inf

        if feasible:
            logger.info(
                f"{rcpsp_max.instance_folder}_PSP{rcpsp_max.instance_id} FEASIBLE "
                f"makespan={result['obj']} sample={duration_sample}"
            )
        else:
            logger.info(
                f"{rcpsp_max.instance_folder}_PSP{rcpsp_max.instance_id} INFEASIBLE "
                f"sample={duration_sample}"
            )

    return result
