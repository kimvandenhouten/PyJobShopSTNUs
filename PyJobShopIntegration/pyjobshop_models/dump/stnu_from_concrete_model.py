from general.logger import get_logger
logger = get_logger(__name__)

"""
Code taken from:
https://pyjobshop.org/latest/examples/flexible_job_shop.html
"""

# FJSP data
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

from pyjobshop import Model

m = Model()

machines = [
    m.add_machine(name=f"Machine {idx}") for idx in range(NUM_MACHINES)
]

jobs = {}
tasks = {}
task_indices = [] # hacky to match (job_idx, task_idx) with each other
for job_idx, job_data in enumerate(data):
    job = m.add_job(name=f"Job {job_idx}")
    jobs[job_idx] = job

    for idx in range(len(job_data)):
        task_idx = (job_idx, idx)
        tasks[task_idx] = m.add_task(job, name=f"Task {task_idx}")
        task_indices.append(task_idx)

for job_idx, job_data in enumerate(data):
    for idx, task_data in enumerate(job_data):
        task = tasks[(job_idx, idx)]

        for duration, machine_idx in task_data:
            machine = machines[machine_idx]
            m.add_mode(task, machine, duration)

    for idx in range(len(job_data) - 1):
        first = tasks[(job_idx, idx)]
        second = tasks[(job_idx, idx + 1)]
        m.add_end_before_start(first, second)

print(f'All tasks must get an start and end node in an STNU')
print(m.tasks)

print(f'Contingent links must be created between start and end of nodes')

print(f'All temporal constraints must be modelled in an STNU')
print(f'End before end {m.constraints.end_before_end}')
print(f'End before start {m.constraints.end_before_start}')
print(f'Start before end {m.constraints.start_before_end}')
print(f'Start before start {m.constraints.start_before_start}')
print(f'Set up times {m.constraints.setup_times}')

print(f'All resource chains must be modelled in an STNU')
result = m.solve(display=False)
from PyJobShopIntegration.pyjobshop_models.utils import find_schedule_per_resource
schedule_per_resource = find_schedule_per_resource(result.best)
print(f'schedule per resource: {schedule_per_resource}')

print(f'I miss the deadlines in the PyJobShop model? They are present in the data though')

