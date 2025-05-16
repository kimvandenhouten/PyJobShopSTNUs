from PyJobShopIntegration.parser import create_instance
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

PATH = "PyJobShopIntegration/data/fjsp/barnes/mt10c1.fjs"
PROBLEM_TYPE = "fjsp"
model = create_instance(PATH, PROBLEM_TYPE)

#
# # FJSP example from PyJobShop documentation
# NUM_MACHINES = 3
#
# # Each job consists of a list of tasks. A task is represented
# # by a list of tuples (processing_time, machine), denoting the eligible
# # machine assignments and corresponding processing times.
# data = [
#     [  # Job with three tasks
#         [(3, 0), (1, 1), (5, 2)],  # task with three eligible machine
#         [(2, 0), (4, 1), (6, 2)],
#         [(2, 0), (3, 1), (1, 2)],
#     ],
#     [
#         [(2, 0), (3, 1), (4, 2)],
#         [(1, 0), (5, 1), (4, 2)],
#         [(2, 0), (1, 1), (4, 2)],
#     ],
#     [
#         [(2, 0), (1, 1), (4, 2)],
#         [(2, 0), (3, 1), (4, 2)],
#         [(3, 0), (1, 1), (5, 2)],
#     ],
# ]
#
# # Construct the models and add the constraints frmo the data
# model = Model()
#
# machines = [
#     model.add_machine(name=f"Machine {idx}") for idx in range(NUM_MACHINES)
# ]
#
# jobs = {}
# tasks = {}
#
# for job_idx, job_data in enumerate(data):
#     job = model.add_job(name=f"Job {job_idx}")
#     jobs[job_idx] = job
#
#     for idx in range(len(job_data)):
#         task_idx = (job_idx, idx)
#         tasks[task_idx] = model.add_task(job, name=f"Task {task_idx}")
#
#
# for job_idx, job_data in enumerate(data):
#     for idx, task_data in enumerate(job_data):
#         task = tasks[(job_idx, idx)]
#
#         for duration, machine_idx in task_data:
#             machine = machines[machine_idx]
#             model.add_mode(task, machine, duration)
#
#     for idx in range(len(job_data) - 1):
#         first = tasks[(job_idx, idx)]
#         second = tasks[(job_idx, idx + 1)]
#         model.add_end_before_start(first, second)

# Solving
result = model.solve(display=False)
solution = result.best

### HERE STARTS OUR CODE ###
# Define the stochastic processing time distributions
duration_distributions = DiscreteUniformSampler(lower_bounds=np.random.randint(2, 5, len(model.tasks)),
                                 upper_bounds=np.random.randint(6, 11, len(model.tasks)))

# Create stnu from concrete model
stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
logger.info(f'Current number of edges {stnu.count_edges()}')

# Add resource chains from solution to the stnu
stnu.add_resource_chains(solution, model)

# Write stnu to xml which is required for using Java CSTNU tool algorithms
stnu_to_xml(stnu, f"example_from_pyjobshop", "temporal_networks/cstnu_tool/xml_files")

# Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files",
                                       f"example_from_pyjobshop")

if dc:
    logger.info(f'The network resulting from the PyJobShop solution is DC')
else:
    logger.info(f'The network is not DC')

if dc:
    # TODO: we could have some sort of Simulator/Evaluator class to do all of this
    # Read ESTNU xml file into Python object that was the output from the previous step
    estnu = STNU.from_graphml(output_location)
    sample_duration = duration_distributions.sample()
    sample = sample_for_rte(sample_duration, estnu)  # TODO: this could then be integrated in a Simulator Class
    logger.debug(f'Sample dict that will be given to RTE star is {sample_duration}')

    # Run RTE algorithm with sample
    rte_data = rte_star(estnu, oracle="sample", sample=sample)

    # Convert to PyJobShop solution for visualization
    ## TODO: currently objective is not overwritten in Solution object
    simulated_solution, objective = rte_data_to_pyjobshop_solution(solution, estnu, rte_data, len(model.tasks), "makespan")
    logger.info(f'Simulated solution has objective {objective}')
    plot_machine_gantt(simulated_solution, model.data(), plot_labels=True)
    plt.savefig('PyJobShopIntegration/images/fjsp_example.png')







