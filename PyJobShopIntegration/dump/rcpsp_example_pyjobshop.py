"""
Code taken from:
https://pyjobshop.org/latest/examples/project_scheduling.html
"""


from pyjobshop import Model, MAX_VALUE

import re
from dataclasses import dataclass
import os

from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
from PyJobShopIntegration.utils import (rte_data_to_pyjobshop_solution, sample_for_rte, get_resource_chains,
                                        add_resource_chains)
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler

from pyjobshop.Model import Model
from pyjobshop.plot import plot_task_gantt, plot_resource_usage

from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.stnu import STNU
from temporal_networks.rte_star import rte_star
from PyJobShopIntegration.utils import plot_stnu
import numpy as np
from general.logger import get_logger
import matplotlib.pyplot as plt
from typing import NamedTuple
logger = get_logger(__name__)


class Mode(NamedTuple):
    job: int
    duration: int
    demands: list[int]


@dataclass(frozen=True)
class Instance:
    """
    Problem instance class based on PSPLIB files.

    Code taken from:
    https://alns.readthedocs.io/en/latest/examples/resource_constrained_project_scheduling_problem.html
    """

    num_jobs: int  # jobs in RCPSP are tasks in PyJobshop
    num_resources: int
    successors: list[list[int]]
    predecessors: list[list[int]]
    modes: list[Mode]
    capacities: list[int]
    renewable: list[bool]
    deadlines: dict[int, int]

    @classmethod
    def read_instance(cls, path: str) -> "Instance":
        """
        Reads an instance of the RCPSP from a file.
        Assumes the data is in the PSPLIB format.
        """
        with open(path) as fh:
            lines = fh.readlines()

        prec_idx = lines.index("PRECEDENCE RELATIONS:\n")
        req_idx = lines.index("REQUESTS/DURATIONS:\n")
        avail_idx = lines.index("RESOURCEAVAILABILITIES:\n")
        deadlines_idx = lines.index("DEADLINES:\n")

        successors = []

        for line in lines[prec_idx + 2 : req_idx - 1]:
            _, _, _, _, *jobs, _ = re.split(r"\s+", line)
            successors.append([int(x) - 1 for x in jobs])

        predecessors: list[list[int]] = [[] for _ in range(len(successors))]
        for job in range(len(successors)):
            for succ in successors[job]:
                predecessors[succ].append(job)

        mode_data = [
            re.split(r"\s+", line.strip())
            for line in lines[req_idx + 3 : avail_idx - 1]
        ]

        # Prepend the job index to mode data lines if it is missing.
        for idx in range(len(mode_data)):
            if idx == 0:
                continue

            prev = mode_data[idx - 1]
            curr = mode_data[idx]

            if len(curr) < len(prev):
                curr = prev[:1] + curr
                mode_data[idx] = curr

        modes = []
        for mode in mode_data:
            job_idx, _, duration, *consumption = mode
            demands = list(map(int, consumption))
            modes.append(Mode(int(job_idx) - 1, int(duration), demands))

        _, *avail, _ = re.split(r"\s+", lines[avail_idx + 2])
        capacities = list(map(int, avail))

        renewable = [
            x == "R"
            for x in lines[avail_idx + 1].strip().split(" ")
            if x in ["R", "N"]  # R: renewable, N: non-renewable
        ]

        deadlines = {
            int(line.split()[0]) - 1: int(line.split()[1])
            for line in lines[deadlines_idx + 2 : -1]
        }

        return Instance(
            int(job_idx),
            len(capacities),
            successors,
            predecessors,
            modes,
            capacities,
            renewable,
            deadlines,
        )
directory = os.path.join("PyJobShopIntegration", "data", "mmrcpspwd", "j10")
filename = "j102_2.mm"
instance = Instance.read_instance(os.path.join(directory, filename))
model = Model()

# It's not necessary to define jobs, but it will add coloring to the plot.
jobs = [model.add_job() for _ in range(instance.num_jobs)]
tasks = [
    model.add_task(job=jobs[idx], latest_end=instance.deadlines[idx] if idx in instance.deadlines else MAX_VALUE)
    for idx in range(instance.num_jobs)
]
# resources = [model.add_renewable(capacity) for capacity in instance.capacities]
resources = [
    model.add_renewable(capacity) if instance.renewable[idx] else model.add_non_renewable(capacity)
    for idx, capacity in enumerate(instance.capacities)
]
for idx, duration, demands in instance.modes:
    model.add_mode(tasks[idx], resources, duration, demands)

for idx in range(instance.num_jobs):
    task = tasks[idx]

    for pred in instance.predecessors[idx]:
        model.add_end_before_start(tasks[pred], task)

    for succ in instance.successors[idx]:
        model.add_end_before_start(task, tasks[succ])

result = model.solve(time_limit=5, display=False)
solution = result.best
print(solution.tasks)

### HERE STARTS OUR CODE ###
# Define the stochastic processing time distributions
duration_distributions = DiscreteUniformSampler(lower_bounds=np.random.randint(1, 3, len(model.tasks)),
                                 upper_bounds=np.random.randint(5, 8, len(model.tasks)))
# Create stnu from concrete model
stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)

# TODO: the get_resource_chains does not work yet for multi-mode, and also in general, we have nothing for models
#       with mixed renewable resources and machines
result_tasks = result.best.tasks
schedule_pyjobshop = [{"task": i, "start": task.start, "end": task.end, "mode": task.mode} for i, task in enumerate(result_tasks)]
needs = []
for i, task in enumerate(result_tasks):
    mode = task.mode
    needs.append(instance.modes[mode].demands)
print("Needs: ", needs)
resource_chains, resource_assignments = get_resource_chains(schedule_pyjobshop, instance.capacities, needs,
                                                            complete=True)
stnu = add_resource_chains(stnu, resource_chains)
# Write stnu to xml for DC-checking
file_name = f"example_rcpsp_max_pyjobshop_stnu"
stnu_to_xml(stnu, file_name, "temporal_networks/cstnu_tool/xml_files")

# Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", file_name)
print("Checking DC: ", dc)
if dc:
    logger.info(f'The network resulting from the PyJobShop solution is DC')
else:
    logger.info(f'The network is not DC')

if dc:
    print("Instance is DC")
    # TODO: we could have some sort of Simulator/Evaluator class to do all of this
    # Read ESTNU xml file into Python object that was the output from the previous step
    estnu = STNU.from_graphml(output_location)
    sample_duration = duration_distributions.sample()
    sample = sample_for_rte(sample_duration, estnu)  # TODO: this could then be integrated in a Simulator Class
    logger.debug(f'Sample dict that will be given to RTE star is {sample_duration}')

    # Run RTE algorithm with sample
    rte_data = rte_star(estnu, oracle="sample", sample=sample)
    print("RTE data: ", rte_data)
    # Convert to PyJobShop solution for visualization
    ## TODO: currently objective is not overwritten in Solution object
    simulated_solution, objective = rte_data_to_pyjobshop_solution(solution, estnu, rte_data, len(model.tasks), "makespan")
    logger.info(f'Simulated solution has objective {objective}')

    # Plotting
    data = model.data()
    fig, axes = plt.subplots(
        data.num_resources + 1,
        figsize=(12, 16),
        gridspec_kw={"height_ratios": [6] + [1] * data.num_resources},
    )

    plot_task_gantt(result.best, model.data(), ax=axes[0])
    plot_resource_usage(result.best, model.data(), axes=axes[1:])
    plt.savefig('PyJobShopIntegration/images/rcpsp_max_example_mm.png')
