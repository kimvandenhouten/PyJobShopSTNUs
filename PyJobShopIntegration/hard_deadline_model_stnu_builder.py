# model_stnu_builder.py
import copy
import re
import numpy as np
from pyjobshop.Model import Model
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from PyJobShopIntegration.simulator import Simulator
from temporal_networks.stnu import STNU
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
import general.logger

# Initialize logger
logger = general.logger.get_logger(__name__)
def get_distribution_bounds(model, data, variation: float):
    """
    For each real task Task (j,i):
      - lb = nominal minimal duration
      - ub = ceil(maximal_duration * (1 + variation))
    Dummy‐deadline tasks get [1,1].
    """
    import re
    lbs, ubs = [], []
    for t in model.tasks:
        m = re.match(r"Task \(\s*(\d+)\s*,\s*(\d+)\s*\)", t.name)
        if m:
            j, i = map(int, m.groups())
            ds = [d for _, d in data[j][i]]
            nominal = min(ds)
            maximum = max(ds)
            lb = max(1, int(np.ceil(nominal * (1 - variation))))
            ub = int(np.ceil(maximum * (1 + variation)))
        else:
            # dummy tasks
            lb, ub = 1, 1
        lbs.append(lb)
        ubs.append(ub)

    # debug print
    print(f"  var={variation:.2f} → lb[:5]={lbs[:5]}, ub[:5]={ubs[:5]}")

    return DiscreteUniformSampler(
        lower_bounds=np.array(lbs, dtype=int),
        upper_bounds=np.array(ubs, dtype=int)
    )

def build_cp_and_solve(data, NUM_MACHINES, delta):
    """
    Build the CP model with hard‐deadlines (via dummy tasks), solve it,
    and return (model, solution, tasks_dict, job_deadlines, cp_feasible_flag).
    """
    # compute per‐job deadlines
    # (you can precompute lb_sum_per_job outside and pass it in;
    #  here we assume data is same shape and delta is added to minimal sums)
    # but for clarity we recompute:
    lb_sum_per_job = {
        j: sum(min(d for _, d in data[j][t]) for t in range(len(data[j])))
        for j in range(len(data))
    }
    job_deadlines = { j: lb_sum_per_job[j] + delta for j in lb_sum_per_job }

    # build model
    model = Model()
    model.set_objective(
        weight_makespan=1,
    )

    # machines + deadline‐resource
    machines = [model.add_machine(f"Machine {m}") for m in range(NUM_MACHINES)]
    deadline_res = model.add_renewable(capacity=999, name="DeadlineResource")

    # tasks and precedence
    tasks = {}
    for j, job_data in enumerate(data):
        job = model.add_job(name=f"Job {j}", due_date=job_deadlines[j])
        for i, opts in enumerate(job_data):
            t = model.add_task(job=job, name=f"Task ({j},{i})")
            tasks[(j,i)] = t
            for m,d in opts:
                model.add_mode(t, machines[m], d)
        # chain within‐job
        for i in range(len(job_data)-1):
            model.add_end_before_start(tasks[(j,i)], tasks[(j,i+1)])
        # dummy deadline task
        last = tasks[(j, len(job_data)-1)]
        dtask = model.add_task(
            name=f"Deadline for Job {j}",
            earliest_start=0,
            latest_end=job_deadlines[j]
        )
        model.add_mode(dtask, deadline_res, duration=1)
        model.add_end_before_start(last, dtask)

    # solve
    res = model.solve(solver="cpoptimizer", display=False)
    feas = int(res.status.name in ("FEASIBLE","OPTIMAL"))
    sol = res.best if feas else None

    return model, sol, tasks, job_deadlines, feas

def build_stnu_and_check(model, sol, tasks, job_deadlines, data, variation, xml_folder = "temporal_networks/cstnu_tool/xml_files"):
    """
    Build the STNU from model+solution, check dynamic controllability via CSTNU tool,
    then do one RTE* run to catch any runtime errors. Return True iff both DC and
    the one-shot simulation succeed.
    """
    # 1) build your duration sampler
    sampler = get_distribution_bounds(model, data, variation)

    # 2) build the STNU
    stnu = PyJobShopSTNU.from_concrete_model(model, sampler)
    stnu.add_resource_chains(sol, model)
    # inject your deadline edges exactly as before
    origin = STNU.ORIGIN_IDX
    for job_idx, deadline in job_deadlines.items():
        last_task = tasks[(job_idx, len(data[job_idx]) - 1)]
        task_index = model.tasks.index(last_task)
        finish_node = stnu.translation_dict_reversed[f"{task_index}_{STNU.EVENT_FINISH}"]
        stnu.set_ordinary_edge(origin, finish_node, deadline)

    # 3) check DC with the Java tool
    stnu_to_xml(stnu, "deadline_stnu", xml_folder)
    dc, _ = run_dc_algorithm(xml_folder, "deadline_stnu")
    if not dc:
        # not even statically controllable
        return False

    # 4) one-shot RTE* simulation to catch hidded errors
    try:
        template_stnu = copy.deepcopy(stnu)
        sim = Simulator(
            model=model,
            stnu=template_stnu,
            solution=sol,
            sampler=sampler,
            objective="makespan"
        )
        summary = sim.run_many(runs=500)
        if summary is None:
            # RTE* said “infeasible sample”
            logger.warning("RTE* produced no schedule for one sample → marking as not controllable.")
            return False
    except Exception:
        logger.exception("Exception during RTE* run → marking as not controllable.")
        return False

    # if we got here, both DC and one-shot simulation passed
    return True