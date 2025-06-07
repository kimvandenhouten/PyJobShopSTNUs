import pandas as pd
from matplotlib import pyplot as plt
from PyJobShopIntegration.parser import parse_data_fjsp
import general.logger
from PyJobShopIntegration.robustness_deadlines import run_one_setting

logger = general.logger.get_logger(__name__)

# -------------------------
# PHASE 1: Load instance
# -------------------------
NUM_MACHINES, data = parse_data_fjsp("data/fjsp/fattahi/MFJS8.fjs")
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
# print(lb_sum_nominal)


# ------------------------------------------------------------------
# Soft-deadline vector for the robustness study
#   – we add one global margin (delta_soft) to every job’s nominal sum
#   – no dummy hard-deadline tasks are used here
# ------------------------------------------------------------------
delta_soft = 1100                      # feel free to tweak
job_deadlines = {
    j: lb_sum_nominal[j] + delta_soft
    for j in range(num_jobs)
}

# ------------------------------------------------------------------
# B) Quality & risk profile of the sweet-spot weights (w_e=5, w_t=1)
# ------------------------------------------------------------------
alphas    = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]   # duration-uncertainty factors
sweet_we  = 5
sweet_wt  = 5

records = [
    run_one_setting(sweet_we, sweet_wt, a,
                    data, NUM_MACHINES, job_deadlines, sim_runs=500)
    for a in alphas
]

df_alpha = pd.DataFrame(records)
print(df_alpha)

# ---------------------  Plot   ---------------------
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(df_alpha.alpha, df_alpha.avg_makespan,  'o-', label='E[$C_{max}$]')
ax.plot(df_alpha.alpha, df_alpha.p95_makespan, 's--', label='95-perc $C_{max}$')
ax.plot(df_alpha.alpha, df_alpha.p_tardy*100,  'd-', label='$P$(tardy) [%]')
ax.plot(df_alpha.alpha, df_alpha.p_early*100,  'x--', label='$P$(early) [%]')

ax.set_xlabel(r"Uncertainty factor $\alpha$")
ax.set_ylabel("Time units  /  Percentage")
ax.set_title(f"Robustness profile at $(w_e,w_t)=({sweet_we},{sweet_wt})$")
ax.grid(True); ax.legend(loc="best")
plt.tight_layout()
plt.savefig("images/fjsp_deadlines/quality_vs_alpha.png")
plt.show()