from PyJobShopIntegration.parser import create_instance
from pyjobshop.plot import plot_machine_gantt

import numpy as np
import general.logger
import matplotlib.pyplot as plt

logger = general.logger.get_logger(__name__)

PATH = "PyJobShopIntegration/data/fjsp_sdst/fattahi/Fattahi_setup_01.fjs"
PROBLEM_TYPE = "fjsp"
model = create_instance(PATH, PROBLEM_TYPE, True)

# Solving
result = model.solve(display=False)
solution = result.best
plot_machine_gantt(solution, data = model.data(), plot_labels=True)
plt.tight_layout()
plt.show()