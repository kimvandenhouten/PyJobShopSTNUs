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
from PyJobShopIntegration.pyjobshop_models.utils import find_schedule_per_resource
from temporal_networks.stnu import STNU


class Mode(NamedTuple):
    job: int
    duration: int
    demands: list[int]


# Import
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
from general.logger import get_logger
from rcpsp_max.temporal_networks.stnu_rcpsp_max import RCPSP_STNU, get_resource_chains, add_resource_chains
from PyJobShopIntegration.pyjobshop_models.example_fjsp import PyJobShopSTNU
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

# Create stnu from concrete model
stnu = PyJobShopSTNU.from_concrete_model(model)

# Solve model and get resource chains
result = model.solve(time_limit=5, display=False)
solution = result.best

#schedule_per_resource = find_schedule_per_resource(solution)
#logger.info('What happens now with schedule per resource code {schedule_per_resource}')

# TODO: the get_resource_chains does not yet work generically for PyJobShop models (also doesn't work for multi-mode)
schedule_pyjobshop = [{"task": i, "start": task.start, "end": task.end} for i, task in enumerate(result.best.tasks)]

resource_chains, resource_assignments = get_resource_chains(schedule_pyjobshop, instance.capacity, instance.needs,
                                                            complete=True)
stnu = add_resource_chains(stnu, resource_chains)

# Write stnu to xml for DC-checking
file_name = f"example_rcpsp_max_pyjobshop_stnu"
stnu_to_xml(stnu, file_name, "temporal_networks/cstnu_tool/xml_files")

# Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", file_name)

if dc:
    logger.info(f'The network resulting from the PyJobShop solution is DC')
else:
    logger.info(f'The network is not DC')