import time
import copy
import numpy as np
from pyjobshop import Model, Task

from FJSP import compute_finish_times
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

def run_reactive_online(
    duration_sample: np.ndarray,
    data_dict: dict,
    fjsp_instance,
    result,
    time_limit: float
) -> dict:
    import time, copy
    from pyjobshop import Model, Task

    data     = copy.deepcopy(data_dict)
    data['time_limit'] = time_limit
    orig_pd  = fjsp_instance.model.data()
    pd_tasks = orig_pd.tasks
    num_tasks= orig_pd.num_tasks

    # true & estimated durations per mode
    true_mode_durs = (
        duration_sample
        if isinstance(duration_sample, dict)
        else {m: int(duration_sample[m]) for m in range(len(duration_sample))}
    )
    est_mode_durs = data.get('estimated_durations', {}).copy()
    solver_calls  = data.get('solver_calls', 0)

    # initial plan
    current_res = result
    committed   = {}    # task_idx -> (start_time, end_time)
    completed   = set()
    t0          = time.time()

    while len(completed) < num_tasks:
        # 1) map TaskData → (task_idx, est_start, mode)
        est_start = {}
        task_mode = {}
        for td in current_res.best.tasks:
            m_idx = td.mode
            t_idx = orig_pd.modes[m_idx].task
            est_start[t_idx]  = td.start
            task_mode[t_idx]  = m_idx

        # 2) compute real finish times
        real_fin = {
            i: est_start[i] + true_mode_durs[ task_mode[i] ]
                for i in est_start if i not in committed
        }

        # 3) select next_i
        next_i = min(real_fin, key=real_fin.get)
        t_now  = real_fin[next_i]

        # 4) commit only that one
        st, en = est_start[next_i], t_now
        committed[next_i] = (st, en)
        completed.add(next_i)
        # update its estimated mode duration
        est_mode_durs[ task_mode[next_i] ] = true_mode_durs[ task_mode[next_i] ]

        # 5) build tasks: freeze only next_i
        new_tasks = []
        for idx, ot in enumerate(pd_tasks):
            if idx == next_i:
                nt = Task(
                    job            = ot.job,
                    earliest_start = st, latest_start = st,
                    earliest_end   = en, latest_end   = en,
                    fixed_duration = True,
                    name           = ot.name
                )
            else:
                nt = ot
            new_tasks.append(nt)


        # 6) rebuild & solve
        dur_model  = fjsp_instance.model_new_durations(est_mode_durs)
        pd2        = dur_model.data().replace(tasks=new_tasks)
        model2     = Model.from_data(pd2)
        current_res= model2.solve(
            solver     = 'cpoptimizer',
            time_limit = time_limit,
            display    = False
        )
        solver_calls += 1

        if not current_res or current_res.status not in ('Feasible', 'Optimal'):
            data['feasibility'] = False
            break

    # finalize
    t_online = time.time() - t0
    ok = len(completed) == num_tasks and current_res.status in ('Feasible', 'Optimal')
    if not ok:
        data.update({
            'time_online': float('inf'),
            'time_limit':  time_limit,
            'feasibility': False,
            'obj':         float('inf')
        })
    else:
        data.update({
            'time_online': t_online,
            'solver_calls': solver_calls,
            'time_limit':  time_limit,
            'start_times': [committed[i][0] for i in range(num_tasks)],
            'obj':         max(e for (_,e) in committed.values()),
            'feasibility': True
        })
    return data