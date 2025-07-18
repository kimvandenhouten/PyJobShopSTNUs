import numpy as np
from pyjobshop.Model import Model
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.simulator import Simulator
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from temporal_networks.stnu import STNU

def run_one_setting(w_e: int,
                    w_t: int,
                    alpha: float,
                    data,
                    num_machines: int,
                    job_deadlines: dict[int, int],
                    sim_runs: int = 500):
    """
    Build a soft-deadline CP model (NO dummy hard deadlines),
    solve it, translate to STNU, simulate `sim_runs` draws.

    Returns a dict with aggregated metrics.
    """
    # -------------------------------
    # 1. CP model (soft deadlines)
    # -------------------------------
    model = Model()
    model.set_objective(weight_makespan=1,
                        weight_total_earliness=w_e,
                        weight_total_tardiness=w_t)

    machines = [model.add_machine(f"M{i}") for i in range(num_machines)]
    tasks: dict[tuple[int, int], any] = {}

    for j, job_data in enumerate(data):
        job = model.add_job(name=f"J{j}",  # purely cosmetic
                            due_date=job_deadlines[j])   # used for E/T
        for i, opts in enumerate(job_data):
            t = model.add_task(job, name=f"J{j}_T{i}")
            tasks[(j, i)] = t
            for m, d in opts:
                model.add_mode(t, machines[m], d)
        for i in range(len(job_data) - 1):
            model.add_end_before_start(tasks[(j, i)],
                                       tasks[(j, i + 1)])

    cp_res   = model.solve(solver="cpoptimizer", display=False)
    cp_sol   = cp_res.best
    cp_mspan = cp_res.objective

    # -------------------------------
    # 2. Build STNU
    # -------------------------------
    # build α-dependent sampler
    l_b, u_b = [], []
    for j, job_data in enumerate(data):
        for i, opts in enumerate(job_data):
            d_min = min(d for _, d in opts)
            d_max = max(d for _, d in opts)
            l_b.append(max(1, int(np.floor((1 - alpha) * d_min))))
            u_b.append(int(np.ceil ((1 + alpha) * d_max)))
    sampler = DiscreteUniformSampler(np.array(l_b), np.array(u_b))

    stnu = PyJobShopSTNU.from_concrete_model(model, sampler)
    stnu.add_resource_chains(cp_sol, model)

    # -------------------------------
    # 3. Simulation
    # -------------------------------
    sim = Simulator(model, stnu, cp_sol, sampler, objective="makespan")

    makespans, tardy, early = [], 0, 0
    job_cnt = len(data)

    # pre-compute index of the last task per job in model.tasks
    last_idx = {j: model.tasks.index(tasks[(j, len(data[j]) - 1)])
                for j in range(job_cnt)}

    for _ in range(sim_runs):
        sim_sol, _ = sim.run_once()
        if sim_sol is None:
            continue                       # STNU execution failed
        makespans.append(max(t.end for t in sim_sol.tasks))

        for j in range(job_cnt):
            finish = sim_sol.tasks[last_idx[j]].end
            if finish > job_deadlines[j]:
                tardy += 1
            else:
                early += 1

    n_jobs = len(makespans) * job_cnt
    return {
        "w_e": w_e,
        "w_t": w_t,
        "alpha": alpha,
        "cp_makespan":  cp_mspan,
        "avg_makespan": np.mean(makespans),
        "p95_makespan": np.percentile(makespans, 95),
        "p_tardy":      tardy / n_jobs,
        "p_early":      early / n_jobs,
    }