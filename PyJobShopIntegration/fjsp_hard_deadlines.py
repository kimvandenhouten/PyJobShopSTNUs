import os
import numpy as np
import matplotlib.pyplot as plt

from pyjobshop.Model import Model
from pyjobshop.plot import plot_machine_gantt
from PyJobShopIntegration.utils import rte_data_to_pyjobshop_solution, sample_for_rte
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler

from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.stnu import STNU
from temporal_networks.rte_star import rte_star

import general.logger

# Initialize logger
logger = general.logger.get_logger(__name__)

# -------------------------
# PHASE 1: Problem Definition
# -------------------------

NUM_MACHINES = 3

job_deadlines = {
    0: 30,
    1: 30,
    2: 27,
    3: 27,
    4: 27,
}

data = [
    [
        [(6, 0), (4, 1), (3, 2)],
        [(2, 0), (5, 1), (3, 2)],
        [(3, 0), (6, 1), (5, 2)],
        [(1, 0), (2, 1), (4, 2)],
    ],
    [
        [(6, 0), (3, 1), (4, 2)],
        [(6, 0), (3, 1), (4, 2)],
        [(3, 0), (5, 1), (1, 2)],
        [(2, 0), (1, 1), (3, 2)],
    ],
    [
        [(3, 0), (6, 1), (1, 2)],
        [(6, 0), (3, 1), (6, 2)],
        [(5, 0), (4, 1), (1, 2)],
        [(4, 0), (1, 1), (3, 2)],
    ],
    [
        [(5, 0), (3, 1), (4, 2)],
        [(3, 0), (5, 1), (3, 2)],
        [(3, 0), (1, 1), (2, 2)],
        [(6, 0), (2, 1), (5, 2)],
    ],
    [
        [(6, 0), (2, 1), (2, 2)],
        [(3, 0), (3, 1), (2, 2)],
        [(1, 0), (2, 1), (1, 2)],
        [(4, 0), (3, 1), (3, 2)],
    ],
]

# -------------------------
# PHASE 2: Build and Solve the CP Model
# -------------------------

model = Model()
model.set_objective(
    weight_makespan=1,
    weight_total_earliness=10,
    weight_total_tardiness=0,
    weight_max_lateness=1000,
)

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

# -------------------------
# PHASE 3: STNU Construction
# -------------------------

duration_distributions = DiscreteUniformSampler(
    lower_bounds=np.full(len(model.tasks), 1),
    upper_bounds=np.full(len(model.tasks), 3),
)

stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
stnu.add_resource_chains(solution, model)

# Add deadline constraints into STNU
for job_idx, deadline in job_deadlines.items():
    last_task = tasks[(job_idx, len(data[job_idx]) - 1)]
    task_index = model.tasks.index(last_task)
    finish_node = stnu.translation_dict_reversed[f"{task_index}_{STNU.EVENT_FINISH}"]
    stnu.set_ordinary_edge(finish_node, STNU.ORIGIN_IDX, -deadline)

# -------------------------
# PHASE 4: Check Dynamic Controllability
# -------------------------

os.makedirs("temporal_networks/cstnu_tool/xml_files", exist_ok=True)
stnu_to_xml(stnu, "fjsp_deadlines_stnu", "temporal_networks/cstnu_tool/xml_files")
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", "fjsp_deadlines_stnu")

if dc:
    logger.info("The network is dynamically controllable.")
else:
    logger.warning("The network is NOT dynamically controllable.")

# -------------------------
# PHASE 5: Real-Time Execution Simulation
# -------------------------

if dc:
    estnu = STNU.from_graphml(output_location)
    sample_duration = duration_distributions.sample()
    sample = sample_for_rte(sample_duration, estnu)

    rte_data = rte_star(estnu, oracle="sample", sample=sample)

    simulated_solution, objective = rte_data_to_pyjobshop_solution(
        solution, estnu, rte_data, len(model.tasks), "makespan"
    )

    # Check deadlines
    for job_idx, deadline in job_deadlines.items():
        last_task = simulated_solution.tasks[len(data[job_idx]) * job_idx + len(data[job_idx]) - 1]
        if last_task.end > deadline:
            logger.warning(f"Job {job_idx} missed its deadline! Ended at {last_task.end}, deadline was {deadline}")
        else:
            logger.info(f"Job {job_idx} met its deadline: finished at {last_task.end} / deadline {deadline}")

    # -------------------------
    # PHASE 6: Visualization
    # -------------------------

    plot_machine_gantt(simulated_solution, model.data(), plot_labels=True)
    os.makedirs("PyJobShopIntegration/images", exist_ok=True)
    plt.savefig('PyJobShopIntegration/images/fjsp_with_deadlines.png')
    plt.show()
