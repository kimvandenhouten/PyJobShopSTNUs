import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from FJSP import FJSP
from parser import create_instance

# ——— Configuration ———
dir_path   = Path(__file__).resolve().parent.parent
DATA_ROOT  = dir_path / "data" / "fjsp_sdst" / "fattahi"
PROB_TYPE  = "fjsp_sdst"

# ——— Collect durations ———
records = []
for fn in os.listdir(DATA_ROOT):
    inst_name = os.path.splitext(fn)[0]
    inst_path = DATA_ROOT / fn

    model     = create_instance(str(inst_path), PROB_TYPE, True)
    modelFJSP = FJSP(model)
    durations = modelFJSP.get_durations()  # Just raw durations

    for dur in durations:
        records.append({
            "Instance": inst_name,
            "Duration": dur
        })

# ——— DataFrame ———
df = pd.DataFrame.from_records(records)

instances = sorted(df["Instance"].unique())
data = [df[df.Instance == inst]["Duration"].values for inst in instances]

# ——— Boxplot with numeric x‐labels ———
plt.figure(figsize=(12, 6))
plt.boxplot(data, labels=range(1, len(instances)+1), showfliers=True)

plt.xlabel("Instance Number")
plt.ylabel("Duration")
plt.title("Distribution of Deterministic Task Durations per Instance")
plt.tight_layout()
plt.show()