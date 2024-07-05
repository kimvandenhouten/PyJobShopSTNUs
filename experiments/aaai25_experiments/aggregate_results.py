import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for instance_folder in ["j10", "j20", "j30", "ubo50", "ubo100"]:
    df = pd.read_csv(f'experiments/aaai25_experiments/results/results_proactive_{instance_folder}_quantile_0.9_60_1.csv')
    #print(f'Number of experiments {instance_folder} proactive {len(df)}')
    df = df[df['obj'] != np.inf]
    print(f'Number of feasible instances 0.9 quantile {instance_folder}  {len(df)}')


for instance_folder in ["j10", "j20", "j30", "ubo50", "ubo100"]:
    # Read the CSV files into DataFrames
    df = pd.read_csv(f'experiments/aaai25_experiments/results/results_proactive_{instance_folder}_SAA_smart_180_1.csv')
    #print(f'Number of experiments {instance_folder} proactive {len(df)}')
    df = df[df['obj'] != np.inf]
    print(f'Number of feasible instances SAA smart {instance_folder} robust {len(df)}')

