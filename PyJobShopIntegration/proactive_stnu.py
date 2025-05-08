import time
import copy
import numpy as np
import general.logger

from PyJobShopIntegration.Sampler       import DiscreteUniformSampler
from PyJobShopIntegration.utils import find_schedule_per_resource, rte_data_to_pyjobshop_solution, sample_for_rte
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from pyjobshop.Model                     import Model, Result
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.rte_star          import rte_star
from temporal_networks.stnu              import STNU

logger = general.logger.get_logger(__name__)


def run_proactive_offline_fjsp_stnu(
    model: Model,
    sampler: DiscreteUniformSampler,
    precedences: list[tuple[int,int]],
    job_deadlines: dict[int,int],
    mode: str = "robust",
    time_limit: float = 60.0,
    nb_saa: int = 10,
):
    """
    Phase 1: build a static “robust”/quantile/SAA schedule,
    then wrap it into a STNU with resource chains + deadlines,
    and check dynamic controllability.
    """
    data = {
        "method": f"proactive_fjsp_stnu_{mode}",
        "time_offline": None,
        "estimated_durations": None,
        "start_times": None,
        "dc": None,
        "xml": None,
    }
    t0 = time.time()

    # 1) pick durations
    lb, ub = sampler.get_bounds()
    durations = None
    if mode == "robust":
        durations = ub
    elif mode.startswith("quantile_"):
        p = float(mode.split("_")[1])
        durations = [int(lb[i] + p*(ub[i]-lb[i]+1)-1) for i in range(len(lb))]
    elif mode == "SAA":
        # average SAA
        starts = []
        for _ in range(nb_saa):
            sample = sampler.sample()
            res: Result = model.solve(
                sample, solver="ortools", time_limit=time_limit, display=False
            )
            if res.status.name not in ("OPTIMAL","FEASIBLE"):
                raise RuntimeError("SAA sub‐solve failed")
            starts.append([t.start for t in res.best.tasks])
        avg = np.mean(starts,axis=0).astype(int).tolist()
        data["start_times"] = avg
    elif mode == "SAA_smart":
        quants = [0.1,0.25,0.5,0.75,0.9]
        starts = []
        for p in quants:
            vec = [int(lb[i] + p*(ub[i]-lb[i]+1)-1) for i in range(len(lb))]
            res: Result = model.solve(
                vec, solver="ortools", time_limit=time_limit, display=False
            )
            if res.status.name not in ("OPTIMAL","FEASIBLE"):
                raise RuntimeError("SAA_smart sub‐solve failed")
            starts.append([t.start for t in res.best.tasks])
        data["start_times"] = np.mean(starts,axis=0).astype(int).tolist()
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    data["estimated_durations"] = durations
    # 2) if we fixed durations, do one solve to get a start vector & solution object
    if durations is not None:
        res: Result = model.solve(
            durations, solver="ortools", time_limit=time_limit, display=False
        )
        if res.status.name not in ("OPTIMAL","FEASIBLE"):
            raise RuntimeError("robust/quantile solve failed")
        data["start_times"] = [t.start for t in res.best.tasks]
        sol = res.best
    else:
        # for SAA modes, re‐solve one of the scenario vectors to get a Solution
        sample = sampler.sample()
        sol = model.solve(
            sample, solver="ortools", time_limit=time_limit, display=False
        ).best

    # 3) build STNU
    stnu = PyJobShopSTNU.from_concrete_model(model, sampler)
    stnu.add_resource_chains(sol, model)
    stnu.add_deadline_constraints(job_deadlines, {
        (j,i): model.tasks.index(t)
        for j,i,t in [((j,i),model.tasks[ model.tasks.index(t) ])
                      for j,i in precedences]
    })
    xml_folder = "temporal_networks/cstnu_tool/xml_files"
    stnu_to_xml(stnu, "proactive_fjsp", xml_folder)
    dc, xml = run_dc_algorithm(xml_folder, "proactive_fjsp")

    data["time_offline"] = time.time() - t0
    data["dc"]           = dc
    data["xml"]          = xml
    data["stnu"]         = stnu
    return data


def run_proactive_online_fjsp_stnu(
    offline: dict,
    sampler: DiscreteUniformSampler,
    runs: int = 1000
):
    """
    Phase 2: if DC, run RTE* on the ORIGINAL STNU
    over `runs` samples, collecting makespan+violations.
    """
    assert offline["dc"], "Network not dynamically controllable!"
    stnu = offline["stnu"]
    summary = {"runs": runs, "violations": 0, "makespans": []}

    for k in range(runs):
        sample = sample_for_rte(sampler.sample(), stnu)   # uses stnu._contingent_nodes
        rte_data = rte_star(stnu, oracle="sample", sample=sample)
        if not hasattr(rte_data, "f"):
            continue

        sol, obj = rte_data_to_pyjobshop_solution(
            offline["sol_base"], stnu, rte_data, len(stnu.translation_dict_reversed), "makespan"
        )
        summary["makespans"].append(obj)
        if obj > max(offline["estimated_durations"]):
            summary["violations"] += 1

    return summary
