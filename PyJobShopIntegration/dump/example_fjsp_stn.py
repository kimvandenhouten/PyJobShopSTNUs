from PyJobShopIntegration.dump.PyJobShopSTN import PyJobShopSTN
from PyJobShopIntegration.Sampler import DiscreteUniformSampler

from pyjobshop.Model import Model
from pyjobshop.plot import plot_machine_gantt

import numpy as np
import general.logger
import matplotlib.pyplot as plt


logger = general.logger.get_logger(__name__)


# FJSP example from PyJobShop documentation
NUM_MACHINES = 3

# Each job consists of a list of tasks. A task is represented
# by a list of tuples (processing_time, machine), denoting the eligible
# machine assignments and corresponding processing times.
data = [
    [  # Job with three tasks
        [(3, 0), (1, 1), (5, 2)],  # task with three eligible machine
        [(2, 0), (4, 1), (6, 2)],
        [(2, 0), (3, 1), (1, 2)],
    ],
    [
        [(2, 0), (3, 1), (4, 2)],
        [(1, 0), (5, 1), (4, 2)],
        [(2, 0), (1, 1), (4, 2)],
    ],
    [
        [(2, 0), (1, 1), (4, 2)],
        [(2, 0), (3, 1), (4, 2)],
        [(3, 0), (1, 1), (5, 2)],
    ],
]

# Construct the models and add the constraints frmo the data
model = Model()

machines = [
    model.add_machine(name=f"Machine {idx}") for idx in range(NUM_MACHINES)
]

jobs = {}
tasks = {}

for job_idx, job_data in enumerate(data):
    job = model.add_job(name=f"Job {job_idx}")
    jobs[job_idx] = job

    for idx in range(len(job_data)):
        task_idx = (job_idx, idx)
        tasks[task_idx] = model.add_task(job, name=f"Task {task_idx}")


for job_idx, job_data in enumerate(data):
    for idx, task_data in enumerate(job_data):
        task = tasks[(job_idx, idx)]

        for duration, machine_idx in task_data:
            machine = machines[machine_idx]
            model.add_mode(task, machine, duration)

    for idx in range(len(job_data) - 1):
        first = tasks[(job_idx, idx)]
        second = tasks[(job_idx, idx + 1)]
        model.add_end_before_start(first, second)

# Solving
result = model.solve(display=False)
solution = result.best

### HERE STARTS OUR CODE ###
# Define the stochastic processing time distributions
duration_distributions = DiscreteUniformSampler(lower_bounds=np.random.randint(2, 5, len(model.tasks)),
                                 upper_bounds=np.random.randint(6, 11, len(model.tasks)))

# Create STN from concrete model
sample_durations = duration_distributions.sample()
pyjobshop_stn = PyJobShopSTN.from_concrete_model(model, sample_durations)

# Add resource chains from solution to the STN
pyjobshop_stn.add_resource_chains(solution, model)

# Add durations to STN
simulated_solution, makespan = pyjobshop_stn.roll_out_schedule(solution)
print(f'makespan is {makespan}')

logger.info(f'Simulated solution has objective {simulated_solution}')
plot_machine_gantt(simulated_solution, model.data(), plot_labels=True)
plt.savefig('PyJobShopIntegration/images/fjsp_example_with_stn.png')








