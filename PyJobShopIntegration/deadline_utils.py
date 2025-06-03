import numpy as np
import re
from PyJobShopIntegration.Sampler import DiscreteUniformSampler


def get_distribution_bounds(model, data):
    """
    Build a DiscreteUniformSampler whose lower_bounds[k], upper_bounds[k]
    correspond exactly to model.tasks[k].

    - For a real Task (named "Task (j, i)"), we look up data[j][i] to get
      its min/max.
    - For any other task (your dummy deadline‐task), we give it a degenerate [1,1].
    """
    lbs = []
    ubs = []
    for t in model.tasks:
        nm = t.name
        # match “Task (j, i)”
        m = re.match(r"Task \(\s*(\d+)\s*,\s*(\d+)\s*\)", nm)
        if m:
            j = int(m.group(1))
            i = int(m.group(2))
            # the data[j][i] is a list of (machine,duration) tuples
            ds = [d for _, d in data[j][i]]
            lbs.append(min(ds))
            ubs.append(max(ds))
        else:
            # dummy deadline‐task
            lbs.append(1)
            ubs.append(1)

    return DiscreteUniformSampler(
        lower_bounds=np.array(lbs, dtype=int),
        upper_bounds=np.array(ubs, dtype=int)
    )

def get_bounds(model, data):
    """
    Build a DiscreteUniformSampler whose lower_bounds[k], upper_bounds[k]
    correspond exactly to model.tasks[k].

    - For a real Task (named "Task (j, i)"), we look up data[j][i] to get
      its min/max.
    - For any other task (your dummy deadline‐task), we give it a degenerate [1,1].
    """
    lbs = []
    ubs = []
    for t in model.tasks:
        nm = t.name
        # match “Task (j, i)”
        m = re.match(r"Task \(\s*(\d+)\s*,\s*(\d+)\s*\)", nm)
        if m:
            j = int(m.group(1))
            i = int(m.group(2))
            # the data[j][i] is a list of (machine,duration) tuples
            ds = [d for _, d in data[j][i]]
            lbs.append(min(ds))
            ubs.append(max(ds))
        else:
            # dummy deadline‐task
            lbs.append(1)
            ubs.append(1)

    return lbs, ubs

