import numpy as np
import re
from PyJobShopIntegration.Sampler import DiscreteUniformSampler


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
            lb = nominal
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

