import pandas as pd
from matplotlib import pyplot as plt
from PyJobShopIntegration.parser import parse_data_fjsp
from PyJobShopIntegration.hard_deadline_model_stnu_builder import (
    build_cp_and_solve,
    build_stnu_and_check,
)
import general.logger

logger = general.logger.get_logger(__name__)

# -------------------------
# PHASE 1: Load instance
# -------------------------
NUM_MACHINES, data = parse_data_fjsp("data/fjsp/kacem/Kacem3.fjs")
num_jobs = len(data)

# precompute nominal sums (not varying with uncertainty)
lb_sum_nominal = {
    j: sum(min(d for _, d in data[j][t]) for t in range(len(data[j])))
    for j in range(num_jobs)
}
ub_sum_nominal = {
    j: sum(max(d for _, d in data[j][t]) for t in range(len(data[j])))
    for j in range(num_jobs)
}

# delta sweep
step = 20
max_delta = 400
deltas = list(range(0, max_delta + 1, step))

# uncertainty levels to test
variations = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]

# We'll collect one record per (variation, delta)
records = []
for variation in variations:
    for delta in deltas:
        # 1) build & solve CP (hard‐deadline model)
        model, sol, tasks, job_deadlines, cp_feasible = build_cp_and_solve(
            data, NUM_MACHINES, delta
        )
        if not cp_feasible:
            # infeasible → both flags = 0
            records.append({
                "variation": variation,
                "delta":     delta,
                "cp_ok":     0,
                "dc_ok":     0,
            })
            continue

        # 2) build STNU & check dynamic controllability
        dc_controllable = build_stnu_and_check(
            model, sol, tasks, job_deadlines, data, variation
        )
        records.append({
            "variation": variation,
            "delta":     delta,
            "cp_ok":     1,
            "dc_ok":     1 if dc_controllable else 0,
        })

# build DataFrame
df = pd.DataFrame(records)

# compute critical Δ* (smallest delta with cp_ok=1 & dc_ok=1) for each variation
crit = (
    df[(df.cp_ok == 1) & (df.dc_ok == 1)]
      .groupby("variation")["delta"]
      .min()
      .reset_index(name="delta_star")
)

# -------------------------
# Panel plots: one row per variation, two columns (CP / STNU)
# -------------------------
nvar = len(variations)
fig, axes = plt.subplots(nvar, 2, figsize=(12, 4 * nvar), sharex=True)

for i, variation in enumerate(variations):
    sub = df[df.variation == variation]
    # CP feasible
    ax = axes[i, 0]
    ax.plot(sub.delta, sub.cp_ok, "o-")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Feasible (1)/Infeasible (0)")
    ax.set_title(f"CP Feasible (var={variation})")
    ax.grid(True)

    # STNU controllable
    ax = axes[i, 1]
    ax.plot(sub.delta, sub.dc_ok, "o-")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"STNU Controllable (var={variation})")
    ax.grid(True)

plt.tight_layout()
plt.savefig("images/fjsp_deadlines/feas_vs_delta_by_variation.png")
plt.show()

# -------------------------
# Summary plot: Slack Δ* vs. duration‐variation
# -------------------------
plt.figure(figsize=(6, 4))
plt.plot(crit.variation, crit.delta_star, "o-")
plt.xlabel("Duration variation")
plt.ylabel("Critical Δ*")
plt.title("Slack Δ* required vs. uncertainty")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/fjsp_deadlines/delta_star_vs_variation.png")
plt.show()