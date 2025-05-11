from matplotlib import pyplot as plt

from PyJobShopIntegration.utils import get_project_root
import pandas as pd
from general.logger import get_logger
from scipy.stats import wilcoxon
from itertools import combinations
import numpy as np

logger = get_logger(__name__)

def evaluate_results(now):
    file = get_project_root() / "PyJobShopIntegration" / "results" / f"final_results_{now}.csv"
    logger.info(f"Evaluating results from {file}")
    df = pd.read_csv(file)
    print(df[df['method'] == 'stnu'].head(10))
    df_stnu = df[df['method'] == 'stnu']
    df_proactive = df[df['method'] == 'proactive'].reset_index(drop=True)
    df_reactive = df[df['method'] == 'reactive'].reset_index(drop=True)

    evaluate_method(df_stnu)
    evaluate_method(df_proactive)
    evaluate_method(df_reactive)
    summarize_feasibility(df)
    wilcoxon_test(df)

def evaluate_method(df_method):
    # Group by instance_folder
    for folder, group in df_method.groupby('instance_folder'):
        # Filter out inf and NaN values
        obj = pd.to_numeric(group['obj'], errors='coerce')
        obj = obj[np.isfinite(obj)]

        time_online = pd.to_numeric(group['time_online'], errors='coerce')
        time_online = time_online[np.isfinite(time_online)]

        time_offline = pd.to_numeric(group['time_offline'], errors='coerce')
        time_offline = time_offline[np.isfinite(time_offline)]

        feasibility = pd.to_numeric(group['feasibility'], errors='coerce')
        feasibility = feasibility[np.isfinite(feasibility)]

        average_makespan = obj.mean()
        variance_makespan = obj.var()

        average_online_time = time_online.mean()
        variance_online_time = time_online.var()

        average_offline_time = time_offline.mean()
        variance_offline_time = time_offline.var()

        feasibility_ratio = feasibility.mean()

        # TODO save this as latex to a file
        print(f"Instance Folder: {folder}")
        print(f"  Average makespan: {average_makespan}")
        print(f"  Variance makespan: {variance_makespan}")
        print(f"  Average online time: {average_online_time}")
        print(f"  Variance online time: {variance_online_time}")
        print(f"  Average offline time: {average_offline_time}")
        print(f"  Variance offline time: {variance_offline_time}")
        print(f"  Feasibility ratio: {feasibility_ratio}")

def summarize_feasibility(df):
    feasibility_summary = df.groupby(['method', 'instance_folder'])['feasibility'].agg(['count', 'sum'])
    feasibility_summary['ratio'] = feasibility_summary['sum'] / feasibility_summary['count']
    print("\n=== Feasibility Summary ===")
    print(feasibility_summary)

def wilcoxon_test(df):
    # Pivot using instance_folder as the identifier
    pivot_df = df.pivot_table(index='instance_folder', columns='method', values='obj', aggfunc='mean').dropna()
    methods = df['method'].unique()

    print("\n=== Wilcoxon Signed-Rank Test Results ===")
    for method1, method2 in combinations(methods, 2):
        try:
            stat, p = wilcoxon(pivot_df[method1], pivot_df[method2])
            print(f"{method1} vs {method2} → W={stat:.3f}, p={p:.4f}")
        except ValueError as e:
            print(f"{method1} vs {method2} → Error: {e}")


evaluate_results("05_10_2025,10_57")