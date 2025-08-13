import time
import copy
import numpy as np

from PyJobShopIntegration.FJSP_NW_AND_GTL.FJSP import compute_finish_times
from general.logger import get_logger

logger = get_logger(__name__)


def run_reactive_offline(fjsp_instance, noise, time_limit_initial=60, mode="mean"):
    start_offline = time.time()

    # 1) choose durations-per-mode based on mode
    if mode == "mean":
        durations = fjsp_instance.get_durations()
    elif mode == "robust":
        lb, ub = fjsp_instance.get_bounds(noise_factor=noise)
        durations = ub
    elif mode.startswith("quantile_"):
        lb, ub = fjsp_instance.get_bounds(noise_factor=noise)
        p = float(mode.split("_")[1])
        durations = [int(lb[i] + p * (ub[i] - lb[i] + 1) - 1)
                     for i in range(len(lb))]
    else:
        raise NotImplementedError(f"Unknown mode '{mode}'")

    # 2) initialize data record
    data_dict = {
        'noise_factor': noise,
        'method': f'reactive_{mode}',
        'time_limit_initial': time_limit_initial,
        'time_offline': np.inf,
        'time_online': np.inf,
        'solver_calls': 0,
        'feasibility': False,
        'estimated_durations': durations,
        'estimated_start_times': None,
        'real_durations': None,
        'start_times': None,
        'obj': np.inf,
    }

    # 3) solve initial schedule
    logger.debug(f"Reactive offline: solving with durations {durations}")
    model = fjsp_instance.model_new_durations(durations)
    result = model.solve(solver='cpoptimizer', time_limit=time_limit_initial, display=False)
    data_dict['solver_calls'] = 1

    if result:
        # extract start times per task index
        start_times = [task.start for task in result.best.tasks]
        data_dict['estimated_start_times'] = start_times
        data_dict['time_offline'] = time.time() - start_offline
        data_dict['feasibility'] = True
    else:
        logger.warning("Initial reactive schedule infeasible or timeout.")

    return data_dict, result


def run_reactive_online(fjsp_instance, duration_sample, data_dict, result, time_limit_rescheduling):
    # copy metadata
    data = copy.deepcopy(data_dict)
    data['real_durations'] = str(duration_sample)
    data['time_limit'] = time_limit_rescheduling

    # prerequisites
    est_starts = data.get('estimated_start_times')
    if est_starts is None:
        data['feasibility'] = False
        return data

    n_tasks = len(est_starts)
    completed = set()
    scheduled = [-1] * n_tasks
    solver_calls = data['solver_calls']
    durations = data['estimated_durations'].copy()

    t_online_start = time.time()
    current_result = result

    while len(completed) < n_tasks:
        # 1) compute estimated and real finish times
        if fjsp_instance.model.constraints.setup_times:
            est_starts_vec, est_fin = compute_finish_times(
                duration_sample=durations,
                task_data_list=current_result.best.tasks,
                modes=fjsp_instance.model.modes,
                n_tasks=n_tasks,
                setup_times={(st.machine, st.task1, st.task2): st.duration
                             for st in fjsp_instance.model.constraints.setup_times}
            )
            _, real_fin = compute_finish_times(
                duration_sample=duration_sample,
                task_data_list=current_result.best.tasks,
                modes=fjsp_instance.model.modes,
                n_tasks=n_tasks,
                setup_times={(st.machine, st.task1, st.task2): st.duration
                             for st in fjsp_instance.model.constraints.setup_times}
            )
        else:
            est_fin = [est_starts[i] + durations[i] for i in range(n_tasks)]
            real_fin = [est_starts[i] + duration_sample[i] for i in range(n_tasks)]

        # 2) next completion
        next_i = min((i for i in range(n_tasks) if i not in completed),
                     key=lambda i: real_fin[i])
        t_now = real_fin[next_i]
        completed.add(next_i)

        # 3) fix all started tasks
        for i, st in enumerate(est_starts):
            if st < t_now:
                scheduled[i] = st

        # 4) check deviation
        if est_fin[next_i] != real_fin[next_i]:
            # update durations per mode: map each mode's .task to real duration
            updated = durations.copy()
            for m, mode in enumerate(fjsp_instance.model.modes):
                updated[m] = duration_sample[mode.task]
            durations = updated

            # rebuild and re-solve
            logger.debug(f"Rescheduling at t={t_now} with durations {durations}")
            new_model = fjsp_instance.model_new_durations(durations)
            current_result = new_model.solve(
                solver='cpoptimizer',
                time_limit=time_limit_rescheduling,
                initial_solution=current_result.best,
                display=False,
                trace_log = False
            )
            solver_calls += 1
            if not current_result:
                data['feasibility'] = False
                logger.info("Reactive online: reschedule infeasible.")
                break
            est_starts = [task.start for task in current_result.best.tasks]

    time_online = time.time() - t_online_start

    # finalize
    if completed != set(range(n_tasks)):
        # infeasible during online
        data['time_online'] = np.inf
        data['feasibility'] = False
        data['obj'] = np.inf
    else:
        data['time_online'] = time_online
        data['solver_calls'] = solver_calls
        data['start_times'] = est_starts
        data['feasibility'] = True
        # final makespan = max real finish
        data['obj'] = max(real_fin)

    return data