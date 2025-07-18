from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from PyJobShopIntegration.parser import create_instance
from pyjobshop.plot import plot_machine_gantt

import numpy as np
import general.logger
import matplotlib.pyplot as plt

from PyJobShopIntegration.utils import sample_for_rte, rte_data_to_pyjobshop_solution
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU

logger = general.logger.get_logger(__name__)

PATH = "PyJobShopIntegration/data/fjsp/fattahi/MFJS1.fjs"
PROBLEM_TYPE = "fjsp"
model = create_instance(PATH, PROBLEM_TYPE, False)


# Solving
result = model.solve(solver='cpoptimizer', display=False)
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
stnu_to_xml(stnu, f"fjsp_no_wait_example", "temporal_networks/cstnu_tool/xml_files")

# Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files",
                                       f"fjsp_no_wait_example")

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
    plt.savefig('PyJobShopIntegration/images/fjsp_no_wait_example.png')