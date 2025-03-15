"""
Code inspired by:
https://pyjobshop.org/latest/examples/project_scheduling.html

This code is used to replace the CP model used in the AAAI paper with a PyJobShop variants.
If we would continue with this code, improvements can be in the direction of:
- Directly using the objects from PyJobShop to do the resource chaining (now there is ugly step to a dict)
- Directly using the constraints from the PyJobShop model to construct the constraints in the Temporal Network
  (now this is coded in rcpsp_max/temporal_networks/stnu_rcpsp_max.py)
"""

from pyjobshop import Model
from typing import NamedTuple



class Mode(NamedTuple):
    job: int
    duration: int
    demands: list[int]


# Import
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
from general.logger import get_logger
from rcpsp_max.temporal_networks.stnu_rcpsp_max import RCPSP_STNU, get_resource_chains, add_resource_chains
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
logger = get_logger(__name__)


# GENERAL SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'
instance_folder = "j10"
instance_id = 1
noise_factor = 1

# Data parsching
instance = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor)
res, schedule = instance.solve()
schedule = schedule.to_dict('records')
print(f'Solution obtained by original CP model {res.get_objective_value()}')
print(f'Schedule {schedule}')

# MODEL
model = Model()

# It's not necessary to define jobs, but it will add coloring to the plot.
jobs = [model.add_job() for _ in range(instance.num_tasks)]
tasks = [model.add_task(job=jobs[idx]) for idx in range(instance.num_tasks)]
resources = [model.add_renewable(capacity) for capacity in instance.capacity]

for idx in range(instance.num_tasks):
    model.add_mode(tasks[idx], resources, instance.durations[idx], instance.needs[idx])

for (pred, lag, suc) in instance.temporal_constraints:
    model.add_start_before_start(tasks[pred], tasks[suc], delay=lag)

result = model.solve(time_limit=5, display=False)

# This could all be done within get_resource_chains using the actual PyJobShop functionalities
schedule_pyjobshop = [{"task": i, "start": task.start, "end": task.end} for i, task in enumerate(result.best.tasks)]
print(schedule_pyjobshop)
print(f'Results obtained by PyJobShop CP model {res.get_objective_value()}')
print(result)

resource_chains, resource_assignments = get_resource_chains(schedule_pyjobshop, instance.capacity, instance.needs, complete=True)

stnu = RCPSP_STNU.from_rcpsp_max_instance(instance.durations, instance.temporal_constraints, noise_factor=instance.noise_factor)
stnu = add_resource_chains(stnu, resource_chains)
stnu_to_xml(stnu, f"example_rcpsp_max_stnu", "temporal_networks/cstnu_tool/xml_files")

# Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", f"example_rcpsp_max_stnu")

if dc:
    logger.info(f'The network resulting from the PyJobShop solution is DC')
else:
    logger.info(f'The network is not DC')