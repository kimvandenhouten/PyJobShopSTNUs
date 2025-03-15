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

solution = m.solve(display=False)

# Build an STNU from the FJSP problem and solution
from temporal_networks.stnu import STNU
stnu = STNU(origin_horizon=False)
from PyJobShopIntegration.pyjobshop_models.utils import find_schedule_per_resource

# Find schedule per resource
schedule_per_resource = find_schedule_per_resource(solution.best)

for task_idx, task in enumerate(solution.best.tasks):
    task_start = stnu.add_node(f'{task_idx}_{STNU.EVENT_START}')
    task_finish = stnu.add_node(f'{task_idx}_{STNU.EVENT_FINISH}')

    # TODO: add contingent links now with dummy data but should come from data source
    stnu.add_contingent_link(task_start, task_finish, 2, 10)

# TODO: this is now hard-coded instead of automatically constructed from the constraints in the PyJobShop model
# Set precedence constraints
for job_idx, job_data in enumerate(data):
    for idx in range(len(job_data) - 1):
        first = task_indices.index((job_idx, idx))
        second = task_indices.index((job_idx, idx + 1))

        pred_idx_finish = stnu.translation_dict_reversed[f'{first}_{STNU.EVENT_FINISH}']
        suc_idx_start = stnu.translation_dict_reversed[f'{second}_{STNU.EVENT_START}']

        stnu.set_ordinary_edge(suc_idx_start, pred_idx_finish, 0)

# Set resource chains and task durations based on solutions
logger.info(f'The schedule per resource is {schedule_per_resource}')
# Add resource chains
for machine, sequence in schedule_per_resource.items():
    for i in range(len(sequence)-1):
        first_idx = sequence[i]
        second_idx = sequence[i+1]
        logger.info(f'Add resource chain between task {first_idx} and task {second_idx}')
        # the finish of the predecessor should precede the start of the successor
        pred_idx_finish = stnu.translation_dict_reversed[
            f"{first_idx}_{STNU.EVENT_FINISH}"]  # Get translation index from finish of predecessor
        suc_idx_start = stnu.translation_dict_reversed[
            f"{second_idx}_{STNU.EVENT_START}"]  # Get translation index from start of successor

        # add constraint between predecessor and successor
        stnu.set_ordinary_edge(suc_idx_start, pred_idx_finish, 0)


# Run DC-checking
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
stnu_to_xml(stnu, f"example_fjsp_stnu", "temporal_networks/cstnu_tool/xml_files")

# Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", f"example_fjsp_stnu")

if dc:
    logger.info(f'The network resulting from the PyJobShop solution is DC')
else:
    logger.info(f'The network is not DC')


