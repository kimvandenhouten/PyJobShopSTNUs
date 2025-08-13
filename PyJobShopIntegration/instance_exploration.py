import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from parser import create_instance

SPIKE = 1_000_000

def extract_job_durations(model):
    durations = defaultdict(list)
    for c in model.constraints.setup_times:
        dur = getattr(c, "duration", 0)
        t1, t2 = c.task1, c.task2
        if t1 is not None and t1 != t2:
            job_obj = getattr(t1, "job", t1)
            job_id = getattr(job_obj, "id", str(job_obj))
            durations[job_id].append(dur)
    return durations

def plot_readable_instance(model, output_path='plots'):
    job_durs = extract_job_durations(model)
    jobs = list(job_durs.keys())

    # Compute proportions and finite lists
    prop_inf = {j: np.mean([d==SPIKE for d in job_durs[j]]) for j in jobs}
    prop_fe = {j: 1-prop_inf[j] for j in jobs}
    finite = {j: [d for d in job_durs[j] if d!=SPIKE] for j in jobs}

    # Sort jobs by decreasing infeasible rate (for example)
    jobs_sorted = sorted(jobs, key=lambda j: prop_inf[j], reverse=True)

    # Build figure
    fig, (ax_bar, ax_box) = plt.subplots(1,2, figsize=(10,6),
                                         gridspec_kw={'width_ratios':[1,1.2]})
    fig.patch.set_facecolor('white')

    # --- Left: horizontal stacked bar ---
    y = np.arange(len(jobs_sorted))
    fe = [prop_fe[j] for j in jobs_sorted]
    ie = [prop_inf[j] for j in jobs_sorted]

    ax_bar.barh(y, fe, color='#4C78A8', edgecolor='white', label='Feasible')
    ax_bar.barh(y, ie, left=fe, color='#F58518', edgecolor='white', label='Infeasible')
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(jobs_sorted, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Proportion', fontsize=11)
    ax_bar.set_title('Feasible vs Infeasible', fontsize=12, pad=10)
    ax_bar.legend(loc='lower right', frameon=False, fontsize=9)
    ax_bar.grid(axis='x', linestyle='--', color='gray', alpha=0.4)

    # Annotate percentages
    for i, j in enumerate(jobs_sorted):
        ax_bar.text(fe[i]/2, i, f"{fe[i]*100:.0f}%", va='center', ha='center',
                    color='white', fontsize=9)
        ax_bar.text(fe[i] + ie[i]/2, i, f"{ie[i]*100:.0f}%", va='center', ha='center',
                    color='white', fontsize=9)

    # --- Right: horizontal boxplots ---
    data = [finite[j] for j in jobs_sorted]
    bp = ax_box.boxplot(data, vert=False, labels=jobs_sorted, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops={'color':'black'},
                        boxprops={'facecolor':'#54A24B', 'edgecolor':'black'})
    ax_box.set_xlabel('Setup time', fontsize=11)
    ax_box.set_title('Feasible Setup Time Distribution', fontsize=12, pad=10)
    ax_box.grid(axis='x', linestyle='--', color='gray', alpha=0.4)
    ax_box.invert_yaxis()  # align with bar chart

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    """
    Loop over your models, extract sdst, and draw all three plots per model.
    """
    models = []
    dir_path = Path(__file__).resolve().parent.parent
    DATA_ROOT = os.path.join(dir_path, "data", "fjsp_sdst", "fattahi")
    PROBLEM_TYPE = "fjsp_sdst"
    for file_name in os.listdir(DATA_ROOT):
        instance_name = os.path.splitext(file_name)[0]
        instance_path = os.path.join(DATA_ROOT, file_name)
        model = create_instance(instance_path, PROBLEM_TYPE, PROBLEM_TYPE == "fjsp_sdst")
        models.append(model)
        plot_readable_instance(model)