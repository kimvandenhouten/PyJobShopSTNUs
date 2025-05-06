import os
import numpy as np
import matplotlib.pyplot as plt

from pyjobshop.Model import Model
from pyjobshop.plot import plot_machine_gantt

from PyJobShopIntegration.plot_gantt import plot_simulation_gantt
from PyJobShopIntegration.plot_simulation_stats import plot_simulation_statistics
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

NUM_MACHINES = 3

job_deadlines = {
    0: 95,
    1: 95,
}


data = [
    [  # Job 0 (total min duration = 3+2+3+2 = 10)
        [(3, 0), (5, 1)],    # Task 0
        [(2, 1), (4, 2)],    # Task 1
        [(3, 0), (3, 2)],    # Task 2
        [(2, 1), (4, 2)],    # Task 3
    ],
    [  # Job 1 (total min duration = 4+3+2+2 = 11)
        [(4, 0), (5, 1)],    # Task 0
        [(3, 1), (5, 2)],    # Task 1
        [(2, 0), (3, 2)],    # Task 2
        [(2, 1), (3, 2)],    # Task 3
    ],
]

# -------------------------
# PHASE 2: Build and Solve the CP Model
# -------------------------
infeasible_jobs = check_deadline_feasibility(data, job_deadlines)

if infeasible_jobs:
    print("[WARNING] Infeasible jobs found:")
    for job_idx, needed, deadline in infeasible_jobs:
        print(f" - Job {job_idx}: needs ≥ {needed}, deadline = {deadline}")


model = Model()
model.set_objective(
    weight_makespan=1,
    weight_total_earliness=10,
    weight_total_tardiness=0,
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
    upper_bounds=np.full(len(model.tasks), 9),
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
stnu_to_xml(stnu, "fjsp_deadlines_stnu", "temporal_networks/cstnu_tool/xml_files")
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", "fjsp_deadlines_stnu")

if dc:
    logger.info("The network is dynamically controllable.")
    estnu = STNU.from_graphml(output_location)

    # Initialize simulator
    simulator = Simulator(
        model=model,
        stnu=estnu,
        solution=solution,
        sampler=duration_distributions,
        objective="makespan"
    )

    # Run many simulations
    summary = simulator.run_many(runs=1000)

    logger.info(f"[SIMULATION] Deadline violations in {summary['total_runs']} runs: {summary['violations']}")
    first_solution = summary["first_solution"]
    makespans = summary["makespans"]
    plot_simulation_statistics(
        summary["makespans"],
        summary["violations"],
        summary["total_runs"]
    )

    # Optional: plot Gantt chart of first result
    plot_simulation_gantt(
        simulated_solution=first_solution,
        model=model,
        filename="fjsp_simulated_gantt.png",
        plot_type="machine",  # or "task" if RCPSP
        )
