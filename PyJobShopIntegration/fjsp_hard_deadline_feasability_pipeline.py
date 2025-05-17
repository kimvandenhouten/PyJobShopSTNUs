import os
import sys

import pandas as pd

from PyJobShopIntegration.deadline_utils import get_distribution_bounds

# ensure repo root on path
repo_root = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.insert(0, repo_root)

import numpy as np
from matplotlib import pyplot as plt
from pyjobshop.Model import Model

from PyJobShopIntegration.parser import parse_data_fjsp
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.stnu import STNU
import general.logger

logger = general.logger.get_logger(__name__)

# -------------------------
# PHASE 1: Load instance & problem definition
# -------------------------
NUM_MACHINES, data = parse_data_fjsp("data/fjsp/kacem/Kacem1.fjs")
num_jobs = len(data)

# compute per-job sum of minimal durations
lb_sum_per_job = {
    j: sum(min(d for _, d in data[j][t]) for t in range(len(data[j])))
    for j in range(num_jobs)
}
ub_sum_per_job = {
    j: sum(max(d for _, d in data[j][t]) for t in range(len(data[j])))
    for j in range(num_jobs)
}


# theo_gap = int((duration_distributions.upper_bounds - duration_distributions.lower_bounds).sum())

# -------------------------
# PHASE 3: Deadline Slack Sweep
# -------------------------
# prepare xml output folder
xml_folder = "temporal_networks/cstnu_tool/xml_files"
os.makedirs(xml_folder, exist_ok=True)

# generate delta values
deltas = list(range(0, 100 + 1, max(1, 100 // 5)))

cp_ok = []
dc_ok = []

# helper to build CP + STNU with hard deadlines
for delta in deltas:
    # set each job's deadline = nominal_sum + delta
    job_deadlines = { j: lb_sum_per_job[j] + delta for j in range(num_jobs) }

    # build CP model with dummy deadline tasks
    model = Model()
    model.set_objective(weight_makespan=1, weight_total_earliness=0,
                        weight_total_tardiness=0, weight_max_lateness=0)
    machines = [model.add_machine(name=f"Machine {idx}") for idx in range(NUM_MACHINES)]
    tasks = {}
    deadline_resource = model.add_renewable(capacity=999, name="DeadlineResource")

    for job_idx, job_data in enumerate(data):
        job = model.add_job(name=f"Job {job_idx}", due_date=job_deadlines[job_idx])

        for idx, task_options in enumerate(job_data):
            task_idx = (job_idx, idx)
            task = model.add_task(job=job, name=f"Task {task_idx}")
            tasks[task_idx] = task

            for machine_idx, duration in task_options:
                model.add_mode(task, machines[machine_idx], duration)

        # Add precedence constraints
        for idx in range(len(job_data) - 1):
            model.add_end_before_start(tasks[(job_idx, idx)], tasks[(job_idx, idx + 1)])

        # Deadline constraint via dummy task
        last_task = tasks[(job_idx, len(job_data) - 1)]
        deadline_task = model.add_task(
            name=f"Deadline for Job {job_idx}",
            earliest_start=0,
            latest_end=job_deadlines[job_idx],
        )
        model.add_mode(deadline_task, deadline_resource, duration=1)
        model.add_end_before_start(last_task, deadline_task)

    # solve CP
    res = model.solve(display=False)
    feas = 1 if res.status.name in ("FEASIBLE","OPTIMAL") else 0
    cp_ok.append(feas)
    if not feas:
        dc_ok.append(0)
        continue
    sol = res.best
    duration_distributions = get_distribution_bounds(model, data)

    # build STNU
    stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
    stnu.add_resource_chains(sol, model)
    # inject deadline edges
    origin = STNU.ORIGIN_IDX
    for job_idx, deadline in job_deadlines.items():
        last_task = tasks[(job_idx, len(data[job_idx]) - 1)]
        task_index = model.tasks.index(last_task)
        finish_node = stnu.translation_dict_reversed[f"{task_index}_{STNU.EVENT_FINISH}"]
        origin_node = STNU.ORIGIN_IDX
    # export & DC check
    stnu_to_xml(stnu, "deadline_stnu", xml_folder)
    dc, _ = run_dc_algorithm(xml_folder, "deadline_stnu")
    dc_ok.append(1 if dc else 0)
df = pd.DataFrame({
    "delta": deltas,
    "cp_ok":  cp_ok,
    "dc_ok":  dc_ok
})

print(df)
delta_star = next((d for d, c, dcc in zip(deltas, cp_ok, dc_ok) if c and dcc), None)
print(f"\nComputed Δ* = {delta_star}")

# -------------------------
# PHASE 4: Plotting results
# -------------------------
# critical delta*
try:
    delta_star = next(d for d, c, dcc in zip(deltas, cp_ok, dc_ok) if c and dcc)
except StopIteration:
    delta_star = None
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

# Panel 1: CP Feasibility vs Δ
ax1.plot(deltas, cp_ok, "o-", label="CP Feasible")
ax1.axvline(100, linestyle="--", label="Theoretical gap")
if delta_star is not None:
    ax1.axvline(delta_star, linestyle=":", label="Critical Δ*")
optimal_deadlines = {
  j: ub_sum_per_job[j] + delta_star
  for j in range(num_jobs)
}


# 2a) Print them to the console
print(f"Found Δ* = {delta_star}")
print("Optimal hard‐deadlines per job:")
for j, D in optimal_deadlines.items():
    print(f"  Job {j}: D_j = {D}")

# 2b) Export as JSON
import json
with open("optimal_deadlines.json", "w") as f:
    json.dump(optimal_deadlines, f, indent=2)
print("Wrote optimal_deadlines.json")

# ——— or ———

# 2c) Export as CSV via pandas
import pandas as pd
df_dead = pd.DataFrame.from_dict(optimal_deadlines,
                                 orient="index",
                                 columns=["deadline"])
df_dead.index.name = "job"
df_dead.to_csv("optimal_deadlines.csv")
print("Wrote optimal_deadlines.csv")


ax1.set_title("CP Feasibility vs Δ")
ax1.set_xlabel("Δ (slack)")
ax1.set_ylabel("Feasible (1) / Infeasible (0)")
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True)
ax1.legend(loc="upper left")

# Panel 2: STNU Controllability vs Δ
ax2.plot(deltas, dc_ok, "o-", label="STNU Controllable")
ax2.axvline(100, linestyle="--", label="Theoretical gap")
if delta_star is not None:
    ax2.axvline(delta_star, linestyle=":", label="Critical Δ*")
ax2.set_title("STNU Controllability vs Δ")
ax2.set_xlabel("Δ (slack)")
ax2.set_ylabel("Controllable (1) / Not controllable (0)")
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("images/fjsp_deadlines/delta_star.png")
plt.show()