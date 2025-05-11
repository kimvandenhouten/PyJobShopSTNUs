import os
import numpy as np
from matplotlib import pyplot as plt
from pyjobshop.Model import Model

from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.stnu import STNU


def make_model_with_deadlines(data, job_deadlines, num_machines):
    model = Model()
    # Just minimize makespan
    model.set_objective(weight_makespan=1,
                        weight_total_earliness=0,
                        weight_total_tardiness=0,
                        weight_max_lateness=0)

    machines = [model.add_machine(f"M{i}") for i in range(num_machines)]
    deadline_res = model.add_renewable(capacity=999, name="DeadlineResource")

    tasks = {}
    for j, job_ops in enumerate(data):
        # capture the Job object here
        job = model.add_job(name=f"J{j}", due_date=job_deadlines[j])

        # real tasks
        for t_idx, opts in enumerate(job_ops):
            # pass job=job, not job=j
            t = model.add_task(job=job, name=f"J{j}_T{t_idx}")
            tasks[(j, t_idx)] = t
            for m, d in opts:
                model.add_mode(t, machines[m], d)

        # precedence within job
        for t_idx in range(len(job_ops) - 1):
            model.add_end_before_start(
                tasks[(j, t_idx)], tasks[(j, t_idx + 1)]
            )

        # dummy deadline task of duration=1
        last = tasks[(j, len(job_ops) - 1)]
        d = model.add_task(
            job=None,  # no job association needed
            name=f"Deadline_J{j}",
            earliest_start=0,
            latest_end=job_deadlines[j]
        )
        model.add_mode(d, deadline_res, duration=1)
        model.add_end_before_start(last, d)

    return model, tasks


if __name__ == "__main__":
    # 1) load instance
    from PyJobShopIntegration.parser import parse_data_fjsp
    num_machines, data = parse_data_fjsp("data/fjsp/kacem/Kacem1.fjs")
    num_jobs = len(data)

    # 2) each job’s minimal sum of mins
    lb_sum_per_job = {
        j: sum(min(d for _, d in data[j][t]) for t in range(len(data[j])))
        for j in range(num_jobs)
    }

    print(lb_sum_per_job)

    # 3) build a sampler for all real tasks + dummy tasks
    all_lb, all_ub = [], []
    for job_ops in data:
        for opts in job_ops:
            ds = [d for _, d in opts]
            all_lb.append(min(ds))
            all_ub.append(max(ds))
    lb_nom = np.array(all_lb, int)
    ub_nom = np.array(all_ub, int)
    pad_lb = np.ones(num_jobs, int)
    pad_ub = np.ones(num_jobs, int)

    sampler = DiscreteUniformSampler(
        lower_bounds=np.concatenate([lb_nom, pad_lb]),
        upper_bounds=np.concatenate([ub_nom, pad_ub])
    )

    # 4) prepare XML folder
    xml_folder = "temporal_networks/cstnu_tool/xml_files"
    os.makedirs(xml_folder, exist_ok=True)

    # 5) choose Δ range
    total_lb = int(lb_nom.sum())
    total_ub = int(ub_nom.sum())
    gap = total_ub - total_lb
    deltas = sorted(set(range(0, gap + 100, max(1, (gap + 20) // 20))))

    cp_ok = []
    dc_ok = []

    for delta in deltas:
        # deadlines = minimal job length + delta
        job_deadlines = {j: lb_sum_per_job[j] + delta for j in range(num_jobs)}

        # build & solve CP
        model, tasks = make_model_with_deadlines(data, job_deadlines, num_machines)
        res = model.solve(display=False)
        status = res.status.name
        feas_cp = status in ("OPTIMAL", "FEASIBLE")
        cp_ok.append(int(feas_cp))

        if not feas_cp:
            dc_ok.append(0)
            continue

        sol = res.best

        # STNU + resource chains
        stnu = PyJobShopSTNU.from_concrete_model(model, sampler)
        stnu.add_resource_chains(sol, model)

        # inject deadline edges
        origin = STNU.ORIGIN_IDX
        for j in range(num_jobs):
            last = tasks[(j, len(data[j]) - 1)]
            ti = model.tasks.index(last)
            fn = stnu.translation_dict_reversed[f"{ti}_{STNU.EVENT_FINISH}"]
            stnu.set_ordinary_edge(origin, fn, job_deadlines[j])

        # export & DC‐check
        xml_name = "fjsp_deadlines_stnu"
        stnu_to_xml(stnu, xml_name, xml_folder)
        dc, _ = run_dc_algorithm(xml_folder, xml_name)
        dc_ok.append(int(dc))

    theo_gap = sum(ub_nom) - sum(lb_nom)
    print(f"Theoretical full‐horizon gap  ∑(max–min) = {theo_gap}")

    # find Δ* = first delta where cp_ok=1 and dc_ok=1
    delta_star = None
    for d, cpf, dcc in zip(deltas, cp_ok, dc_ok):
        if cpf == 1 and dcc == 1:
            delta_star = d
            break

    if delta_star is None:
        print("No Δ in your sweep makes both CP‐feasible and STNU‐controllable.")
    else:
        print(f"Critical slack Δ* = {delta_star}")
        print(f"  → Δ* / theoretical gap = {delta_star:.1f} / {theo_gap:.1f} = {delta_star / theo_gap:.2f}")

    theo_gap = sum(ub_nom) - sum(lb_nom)
    delta_star = next((d for d, cpf, dcc in zip(deltas, cp_ok, dc_ok) if cpf and dcc), None)

    # --- now the enriched plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # panel 1: CP feasibility
    ax1.plot(deltas, cp_ok, "o-", label="CP Feasible")
    ax1.axvline(theo_gap, color="C2", linestyle="--", label="Theoretical gap")
    if delta_star is not None:
        ax1.axvline(delta_star, color="C3", linestyle=":", label="Critical Δ*")
    ax1.set_title("CP Feasibility vs Δ")
    ax1.set_xlabel("Δ (slack)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # panel 2: STNU controllability
    ax2.plot(deltas, dc_ok, "o-", color="C1", label="STNU Controllable")
    ax2.axvline(theo_gap, color="C2", linestyle="--")
    if delta_star is not None:
        ax2.axvline(delta_star, color="C3", linestyle=":")
    ax2.set_title("STNU Controllability vs Δ")
    ax2.set_xlabel("Δ (slack)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    ax2.legend(loc="upper left")

    # annotate exact values
    ax2.text(theo_gap, 0.5, f" gap={theo_gap}", color="C2", va="center", ha="right", rotation=90)
    if delta_star is not None:
        ax2.text(delta_star, 0.5, f" Δ*={delta_star}", color="C3", va="center", ha="left", rotation=90)

    plt.tight_layout()
    plt.show()
