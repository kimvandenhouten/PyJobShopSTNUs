import numpy as np
import re
from PyJobShopIntegration.Sampler import DiscreteUniformSampler


def get_distribution_bounds(model, data, variation=0.6):
    lbs = []
    ubs = []
    for t in model.tasks:
        nm = t.name
        m = re.match(r"Task \(\s*(\d+)\s*,\s*(\d+)\s*\)", nm)
        if m:
            j, i = map(int, m.groups())
            ds = [d for _, d in data[j][i]]
            nominal = min(ds)
            lb = max(1, int(np.floor(nominal * (1 - variation))))
            ub = int(np.ceil(nominal * (1 + variation)))
        else:
            # dummy‐deadline tasks
            lb, ub = 1, 1
        lbs.append(lb)
        ubs.append(ub)

    return DiscreteUniformSampler(
        lower_bounds=np.array(lbs, dtype=int),
        upper_bounds=np.array(ubs, dtype=int)
    )


