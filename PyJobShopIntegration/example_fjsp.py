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

PATH = "ex_tue.fjs"
PROBLEM_TYPE = "fjsp"
model = create_instance(PATH, PROBLEM_TYPE)

# Solving
result = model.solve(display=False)
solution = result.best
plot_machine_gantt(solution, data = model.data(), plot_labels=True)
plt.tight_layout()
plt.show()