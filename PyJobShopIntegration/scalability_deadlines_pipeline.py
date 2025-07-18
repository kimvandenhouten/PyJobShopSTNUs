import random
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from PyJobShopIntegration.parser import parse_data_fjsp
from PyJobShopIntegration.hard_deadline_model_stnu_builder import (
    build_cp_and_solve,
    get_distribution_bounds,
)
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.simulator import Simulator
from temporal_networks.stnu import STNU
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
import general.logger
import copy

logger = general.logger.get_logger(__name__)
SEED = 12345
np.random.seed(SEED); random.seed(SEED)
instances = {
    "Kacem1": "data/fjsp/kacem/Kacem1.fjs",
    "Kacem2": "data/fjsp/kacem/Kacem2.fjs",
    "Kacem3": "data/fjsp/kacem/Kacem3.fjs",
    "Kacem4": "data/fjsp/kacem/Kacem4.fjs",
}

variation = 1.0
delta = 350

records = []

for name, path in instances.items():
    NUM_MACHINES, data = parse_data_fjsp(path)
    num_tasks = sum(len(job) for job in data)

    start = time.time()
    model, sol, tasks, job_deadlines, cp_feasible = build_cp_and_solve(data, NUM_MACHINES, delta)
    cp_time = time.time() - start

    if not cp_feasible:
        records.append({
            "instance": name,
            "tasks": num_tasks,
            "cp_time": cp_time,
            "stnu_time": None,
            "dc_time": None,
            "rte_time": None,
            "ok": False,
        })
        continue

    try:
        start = time.time()
        sampler = get_distribution_bounds(model, data, variation)
        stnu = PyJobShopSTNU.from_concrete_model(model, sampler)
        stnu.add_resource_chains(sol, model)
        origin = STNU.ORIGIN_IDX
        for job_idx, deadline in job_deadlines.items():
            last_task = tasks[(job_idx, len(data[job_idx]) - 1)]
            task_index = model.tasks.index(last_task)
            finish_node = stnu.translation_dict_reversed[f"{task_index}_{STNU.EVENT_FINISH}"]
            stnu.set_ordinary_edge(origin, finish_node, deadline)
        stnu_time = time.time() - start

        start = time.time()

        stnu_to_xml(stnu, "deadline_stnu", "temporal_networks/cstnu_tool/xml_files")
        dc, _ = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", "deadline_stnu")
        dc_time = time.time() - start

        start = time.time()
        if dc:
            template_stnu = copy.deepcopy(stnu)
            sim = Simulator(model=model, stnu=template_stnu, solution=sol, sampler=sampler, objective="makespan")
            summary = sim.run_many(runs=500)
            rte_time = time.time() - start
        else:
            rte_time = None

        records.append({
            "instance": name,
            "tasks": num_tasks,
            "cp_time": cp_time,
            "stnu_time": stnu_time,
            "dc_time": dc_time,
            "rte_time": rte_time,
            "ok": bool(dc and summary is not None),
        })
    except Exception as e:
        logger.exception("Error during STNU/DC/RTE")
        records.append({
            "instance": name,
            "tasks": num_tasks,
            "cp_time": cp_time,
            "stnu_time": None,
            "dc_time": None,
            "rte_time": None,
            "ok": False,
        })

# Save results
df_scalability = pd.DataFrame(records)
df_scalability.to_csv("scalability_results.csv", index=False)
print(df_scalability)



df_melted = df_scalability.melt(
    id_vars=["instance", "tasks"],
    value_vars=["cp_time", "stnu_time", "dc_time", "rte_time"],
    var_name="stage",
    value_name="time_sec"
)

# Line plot: Time vs. Task count for each stage
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x="tasks", y="time_sec", hue="stage", marker="o")
plt.title("Computational Time per Stage vs. Number of Tasks")
plt.xlabel("Number of Tasks")
plt.ylabel("Time (s)")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/fjsp_deadlines/scalability_results.png")

# Bar plot: Grouped by instance
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="instance", y="time_sec", hue="stage")
plt.title("Computational Cost by Pipeline Stage per Instance")
plt.xlabel("Instance")
plt.ylabel("Time (s)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("images/fjsp_deadlines/scalability_barplot.png")

df_melted.head()