import os
import time

import numpy as np
from matplotlib import pyplot as plt
from pyjobshop.Model import Model
from pyjobshop.plot import plot_machine_gantt

from PyJobShopIntegration.plot_gantt_and_stats import plot_simulation_statistics
from PyJobShopIntegration.reactive_left_shift import group_shift_solution_resequenced
from PyJobShopIntegration.simulator import Simulator
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from general.deadline_utils import check_deadline_feasibility, compute_slack_weights

from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.stnu import STNU

import general.logger

# Initialize logger
logger = general.logger.get_logger(__name__)

# -------------------------
# PHASE 1: Problem Definition
# -------------------------
NUM_MACHINES = 15

# For each job j, deadline = sum of its tasks’ minimal durations + 10
job_deadlines = {j: 17 for j in range(12)}
# data[j] is a list of tasks for the job j; each task is a list of
# (duration, machine) choices.
data = [
    [[(3, 0), (2, 11)], [(1, 3), (5, 2)], [(4, 1), (1, 9)]],
    [[(2, 1), (5, 3)], [(5, 9), (2, 0)], [(5, 11), (4, 10)]],
    [[(3, 7), (1, 9)], [(2, 12), (4, 14)], [(2, 5), (2, 4)], [(1, 12), (1, 5)]],
    [[(3, 1), (5, 5)], [(1, 4), (4, 12)], [(4, 8), (1, 1)],
     [(5, 8), (3, 4)], [(1, 9), (1, 3)], [(3, 10), (1, 3)]],
    [[(4, 13), (3, 1)], [(3, 7), (2, 10)], [(2, 5), (3, 14)], [(1, 11), (5, 10)]],
    [[(2, 8), (2, 11)], [(3, 7), (5, 6)], [(3, 3), (1, 10)], [(1, 3), (3, 13)]],
    [[(2, 4), (5, 1)], [(3, 14), (2, 11)], [(4, 10), (4, 7)],
     [(2, 2), (2, 4)], [(5, 11), (3, 8)], [(4, 11), (5, 9)]],
    [[(2, 5), (5, 3)], [(1, 7), (1, 1)], [(2, 2), (4, 10)],
     [(4, 9), (4, 1)], [(5, 9), (3, 7)], [(1, 8), (1, 13)]],
    [[(3, 12), (1, 10)], [(2, 4), (4, 6)], [(3, 0), (5, 11)],
     [(5, 12), (1, 2)], [(3, 13), (5, 10)]],
    [[(2, 2), (5, 5)], [(1, 12), (5, 8)], [(1, 5), (1, 7)], [(3, 14), (2, 5)]],
    [[(1, 3), (1, 9)], [(1, 11), (5, 7)], [(2, 12), (4, 2)]],
    [[(5, 4), (4, 8)], [(2, 3), (3, 8)], [(3, 6), (4, 10)], [(4, 14), (1, 8)]],
]

# -------------------------
# PHASE 2: Build and Solve the CP Model
# -------------------------
infeasible_jobs = check_deadline_feasibility(data, job_deadlines)

if infeasible_jobs:
    print("[WARNING] Infeasible jobs found:")
    for job_idx, needed, deadline in infeasible_jobs:
        print(f" - Job {job_idx}: needs ≥ {needed}, deadline = {deadline}")
w_e = 100
w_t = 0
is_soft_deadline = True
model = Model()
model.set_objective(
    weight_makespan=1,
    weight_total_earliness=w_e,
    weight_total_tardiness=w_t,
    weight_max_lateness=1000,
)
weights = compute_slack_weights(data, job_deadlines)

machines = [model.add_machine(name=f"Machine {idx}") for idx in range(NUM_MACHINES)]
tasks = {}
deadline_resource = model.add_renewable(capacity=999, name="DeadlineResource")

for job_idx, job_data in enumerate(data):
    job = model.add_job(name=f"Job {job_idx}", due_date=job_deadlines[job_idx])

    for idx, task_options in enumerate(job_data):
        task_idx = (job_idx, idx)
        task = model.add_task(job=job, name=f"Task {task_idx}")
        tasks[task_idx] = task

        for duration, machine_idx in task_options:
            model.add_mode(task, machines[machine_idx], duration)

    # Add precedence constraints
    for idx in range(len(job_data) - 1):
        model.add_end_before_start(tasks[(job_idx, idx)], tasks[(job_idx, idx + 1)])

    # Deadline constraint via dummy task
    last_task = tasks[(job_idx, len(job_data) - 1)]
    if is_soft_deadline is False:
        deadline_task = model.add_task(
            name=f"Deadline for Job {job_idx}",
            earliest_start=0,
            latest_end=job_deadlines[job_idx],
        )
        model.add_mode(deadline_task, deadline_resource, duration=1)
        model.add_end_before_start(last_task, deadline_task)

# Solve
result = model.solve(display=True)
solution = result.best
print(f"Objective value: {result.objective}")
solution = group_shift_solution_resequenced(solution, model)
print("\n[DEBUG] Tasks after shifting:")
for idx, task in enumerate(solution.tasks):
    task_name = model.tasks[idx].name if idx < len(model.tasks) else f"Task {idx}"
    print(f"{task_name}: start={task.start}, end={task.end}, duration={task.end - task.start}")

# -------------------------
# PHASE 3: STNU Construction
# -------------------------

duration_distributions = DiscreteUniformSampler(
    lower_bounds=np.full(len(model.tasks), 1),
    upper_bounds=np.full(len(model.tasks), 6),
)

# 3. Build the STNU
stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
stnu.add_resource_chains(solution, model)

# Add deadline edges again:
for job_idx, deadline in job_deadlines.items():
    last_task = tasks[(job_idx, len(data[job_idx]) - 1)]
    task_index = model.tasks.index(last_task)

    finish_node = stnu.translation_dict_reversed[f"{task_index}_{STNU.EVENT_FINISH}"]
    origin_node = STNU.ORIGIN_IDX
# -------------------------
# PHASE 4: Check Dynamic Controllability
# -------------------------

os.makedirs("temporal_networks/cstnu_tool/xml_files", exist_ok=True)
stnu_to_xml(stnu, "deadline_stnu", "temporal_networks/cstnu_tool/xml_files")
dc, _ = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", "deadline_stnu")

if not dc:
    logger.warning("The network is NOT dynamically controllable.")
else:
    logger.info("Dynamically controllable — running RTE* on the ORIGINAL STNU")

# -------------------------
# PHASE 5: Real-Time Execution Simulation & Plots
# -------------------------
if dc:
    estnu_for_sim = stnu
    sim = Simulator(model, stnu, solution, duration_distributions, objective="makespan")
    summary = sim.run_many(runs=200)
    logger.info(f"Deadline violations in {summary["total_runs"]} runs: {summary["violations"]}")
    # -------------------------
    # Gantt Chart for First Run
    # -------------------------
    plot_machine_gantt(summary["first_solution"], model.data(), plot_labels=True)
    # -------------------------
    # Statistics Plots
    # -------------------------
    plot_simulation_statistics(summary["makespans"], summary["violations"], summary["total_runs"])




def run_soft_deadline_sweep(
    we_values: list[int],
    wt_values: list[int],
    model_factory,
    data, job_deadlines,
    num_machines: int,
    sampler: DiscreteUniformSampler,
    sim_runs: int = 200
):
    results = []

    for w_e in we_values:
        for w_t in wt_values:
            # 1) Build & solve CP
            model = model_factory()
            model.set_objective(
                weight_makespan=1,
                weight_total_earliness=w_e,
                weight_total_tardiness=w_t,
                weight_max_lateness=0,
            )
            # add machines, tasks, precedences (no hard deadlines)
            machines = [model.add_machine(f"M{i}") for i in range(num_machines)]
            tasks = {}
            for j, job_data in enumerate(data):
                job = model.add_job(name=f"J{j}", due_date=job_deadlines[j])
                for t_idx, opts in enumerate(job_data):
                    t = model.add_task(job, name=f"J{j}_T{t_idx}")
                    tasks[(j, t_idx)] = t
                    for d, m in opts:
                        model.add_mode(t, machines[m], d)
                for t_idx in range(len(job_data)-1):
                    model.add_end_before_start(tasks[(j, t_idx)], tasks[(j, t_idx+1)])

            res = model.solve(display=False)
            cp_time = res.runtime
            sol = res.best
            cp_makespan = max(t.end for t in sol.tasks)

            # 2) Build STNU (no DC check for soft‐deadline sweep)
            stnu = PyJobShopSTNU.from_concrete_model(model, sampler)
            stnu.add_resource_chains(sol, model)

            # 3) Simulate sim_runs times
            sim = Simulator(model, stnu, sol, sampler, objective="makespan")
            sum_msp = 0
            sum_tardy = 0
            sum_earl  = 0
            count_tardy = 0
            count_earl  = 0

            for _ in range(sim_runs):
                sim_sol, _ = sim.run_once()
                # global makespan
                msp = max(t.end for t in sim_sol.tasks)
                sum_msp += msp

                # per‐job earliness/tardiness
                for j in job_deadlines:
                    # find the last real task of job j
                    last_idx = len(data[j]) * j + (len(data[j]) - 1)
                    F = sim_sol.tasks[last_idx].end
                    D = job_deadlines[j]
                    earl = max(0, D - F)
                    tard = max(0, F - D)
                    sum_earl  += earl
                    sum_tardy += tard
                    count_earl  += (earl > 0)
                    count_tardy += (tard > 0)

            results.append({
                "w_e":               w_e,
                "w_t":               w_t,
                "cp_makespan":       cp_makespan,
                "cp_time":           cp_time,
                "avg_makespan":      sum_msp / sim_runs,
                "avg_tardiness":     sum_tardy / (sim_runs * len(job_deadlines)),
                "p_tardy":           count_tardy / (sim_runs * len(job_deadlines)),
                "avg_earliness":     sum_earl / (sim_runs * len(job_deadlines)),
                "p_early":           count_earl  / (sim_runs * len(job_deadlines)),
            })

            logger.info(
                f"[w_e={w_e}] CP_ms={cp_makespan:.1f} t={cp_time:.2f}s "
                f"→ make={results[-1]['avg_makespan']:.1f}, "
                f"p_tardy={results[-1]['p_tardy']:.1%}, "
                f"p_early={results[-1]['p_early']:.1%}"
            )

    return results


if __name__ == "__main__":
    total_tasks = sum(len(job) for job in data)
    sampler = DiscreteUniformSampler(
        lower_bounds=np.ones(total_tasks, int),
        upper_bounds=np.full(total_tasks, 6, int),
    )


    def make_model():
        return Model()


    we_values = [0, 1, 5, 10, 20, 50, 100]
    wt_values = [0,1,5,10,20,50,100]
    sweep = run_soft_deadline_sweep(
        we_values, wt_values, make_model,
        data, job_deadlines,
        NUM_MACHINES, sampler,
        sim_runs=200
    )

    import pandas as pd

    df = pd.DataFrame(run_soft_deadline_sweep(
        we_values, wt_values, make_model, data, job_deadlines, NUM_MACHINES, sampler, sim_runs=200
    ))

    ax = df.plot(
        x="w_e",
        y=["cp_makespan", "avg_makespan"],
        marker="o",
        grid=True,
        title="CP vs. Simulated Makespan"
    )
    ax.set_xlabel("Early‐finish weight (w_e)")
    ax.set_ylabel("Makespan")
    plt.show()

    # Plot probability of tardiness
    ax2 = df.plot(
        x="w_e",
        y="p_tardy",
        marker="x",
        grid=True,
        title="Probability of Tardiness"
    )
    ax2.set_xlabel("Early‐finish weight (w_e)")
    ax2.set_ylabel("P(tardy)")
    plt.show()

    x_col = "avg_makespan"
    y_col = "p_tardy"
    pts = df[[x_col, y_col]].to_numpy()
    is_pareto = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        # check if any other q dominates p
        dominated = np.any(
            np.all(pts <= p, axis=1) &  # q ≤ p in all dims
            np.any(pts < p, axis=1)  # and q < p in at least one
        )
        is_pareto[i] = not dominated

    pareto_df = df[is_pareto].sort_values(x_col)

    plt.figure()
    plt.scatter(df[x_col], df[y_col], label="all points")
    plt.scatter(pareto_df[x_col], pareto_df[y_col],
                edgecolor="k", s=100, label="Pareto front")
    plt.plot(pareto_df[x_col], pareto_df[y_col], linestyle="--", color="k")
    plt.xlabel("Average Makespan")
    plt.ylabel("Probability of Tardiness")
    plt.title("Pareto Front: Makespan vs. P(tardy)")
    plt.legend()
    plt.grid(True)
    plt.show()
