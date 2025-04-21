import os
from PyJobShopIntegration.utils import rte_data_to_pyjobshop_solution, sample_for_rte
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler

from pyjobshop.Model import Model
from pyjobshop.plot import plot_machine_gantt

from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.stnu import STNU
from temporal_networks.rte_star import rte_star

import numpy as np
import general.logger
import matplotlib.pyplot as plt

logger = general.logger.get_logger(__name__)
NUM_MACHINES = 3

job_deadlines = {
    0: 28,
    1: 27,
    2: 30,
}

data = [
    [  # Job 0
        [(2, 0), (3, 1)],
        [(3, 1), (4, 2)],
        [(2, 0), (3, 2)],
        [(4, 2), (3, 1)],
    ],
    [  # Job 1
        [(3, 2), (4, 0)],
        [(3, 0), (2, 1)],
        [(4, 2), (2, 0)],
        [(3, 1), (3, 2)],
    ],
    [  # Job 2
        [(2, 1), (3, 0)],
        [(4, 2), (3, 1)],
        [(3, 0), (2, 2)],
        [(3, 2), (4, 1)],
    ],
]

model = Model()
machines = [model.add_machine(name=f"Machine {idx}") for idx in range(NUM_MACHINES)]
jobs = {}
tasks = {}

for job_idx, job_data in enumerate(data):
    job = model.add_job(name=f"Job {job_idx}")
    jobs[job_idx] = job

    for idx in range(len(job_data)):
        task_idx = (job_idx, idx)
        task = model.add_task(job=job, name=f"Task {task_idx}")
        tasks[task_idx] = task

    for idx, task_data in enumerate(job_data):
        task = tasks[(job_idx, idx)]
        for duration, machine_idx in task_data:
            model.add_mode(task, machines[machine_idx], duration)

    for idx in range(len(job_data) - 1):
        model.add_end_before_start(tasks[(job_idx, idx)], tasks[(job_idx, idx + 1)], delay=0)
    last_task = tasks[(job_idx, len(job_data) - 1)]
    deadline = job_deadlines[job_idx]

    deadline_task = model.add_task(
        name=f"Deadline for Job {job_idx}",
        earliest_start=0,
        latest_end=deadline,
    )
    model.add_mode(deadline_task, machines[0], duration=1)
    model.add_end_before_end(last_task, deadline_task)
    model.set_objective(weight_makespan=1)

result = model.solve(display=False)
solution = result.best

duration_distributions = DiscreteUniformSampler(
    lower_bounds=np.random.randint(2, 4, len(model.tasks)),
    upper_bounds=np.random.randint(5, 7, len(model.tasks))
)

stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
stnu.add_resource_chains(solution, model)

for job_idx, deadline in job_deadlines.items():
    last_task = tasks[(job_idx, len(data[job_idx]) - 1)]
    task_index = model.tasks.index(last_task)

    finish_node = stnu.translation_dict_reversed[f"{task_index}_{STNU.EVENT_FINISH}"]
    origin_node = STNU.ORIGIN_IDX

    stnu.set_ordinary_edge(finish_node, origin_node, -deadline)


os.makedirs("temporal_networks/cstnu_tool/xml_files", exist_ok=True)
stnu_to_xml(stnu, f"fjsp_deadlines_stnu", "temporal_networks/cstnu_tool/xml_files")

dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", "fjsp_deadlines_stnu")

if dc:
    logger.info(f"The network is dynamically controllable.")
else:
    logger.info(f"The network is NOT dynamically controllable.")
if dc:
    estnu = STNU.from_graphml(output_location)
    sample_duration = duration_distributions.sample()
    sample = sample_for_rte(sample_duration, estnu)

    rte_data = rte_star(estnu, oracle="sample", sample=sample)

    simulated_solution, objective = rte_data_to_pyjobshop_solution(
        solution, estnu, rte_data, len(model.tasks), "makespan"
    )

    for job_idx, deadline in job_deadlines.items():
        last_task = simulated_solution.tasks[len(data[job_idx]) * job_idx + len(data[job_idx]) - 1]
        if last_task.end > deadline:
            logger.warning(f"Job {job_idx} missed its deadline! Ended at {last_task.end}, deadline was {deadline}")
        else:
            logger.info(f"Job {job_idx} met its deadline: finished at {last_task.end} / deadline {deadline}")

    plot_machine_gantt(simulated_solution, model.data(), plot_labels=True)
    os.makedirs("PyJobShopIntegration/images", exist_ok=True)
    plt.savefig('PyJobShopIntegration/images/fjsp_with_deadlines.png')
