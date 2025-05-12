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
    df_stnu = df[df['method'] == 'stnu']
    df_proactive = df[df['method'].str.startswith('proactive')]
    df_reactive = df[df['method'] == 'reactive']

    evaluate_methods(df)
    summarize_feasibility(df)
    wilcoxon_test(df)


def evaluate_methods(df):
    for method in df['method'].unique():
        print(f"\n=== Evaluating method: {method} ===")
        method_df = df[df['method'] == method]
        # 1) Convert relevant columns to numeric, safely
        method_df['obj'] = pd.to_numeric(method_df['obj'], errors='coerce')
        method_df['time_online'] = pd.to_numeric(method_df['time_online'], errors='coerce')
        method_df['time_offline'] = pd.to_numeric(method_df['time_offline'], errors='coerce')
        method_df['feasibility'] = pd.to_numeric(method_df['feasibility'], errors='coerce')
        # 2) Drop invalid rows
        mask = (
            np.isfinite(method_df['obj']) &
            np.isfinite(method_df['time_online']) &
            np.isfinite(method_df['time_offline']) &
            np.isfinite(method_df['feasibility'])
        )
        method_df = method_df[mask]
        # 3) Group and aggregate
        summary = (
            method_df
            .groupby(['instance_folder'])
            .agg(
                avg_makespan = ('obj', 'mean'),
                var_makespan = ('obj', 'var'),
                avg_online_time = ('time_online', 'mean'),
                var_online_time = ('time_online', 'var'),
                avg_offline_time = ('time_offline', 'mean'),
                var_offline_time = ('time_offline', 'var'),
                feasibility_ratio = ('feasibility', 'mean'),
            )
            .reset_index()  # ← make method & folder columns again
        )
        print("\n=== Method Evaluation Summary ===\n")
        # 4) Loop over every row and print it
        for _, row in summary.iterrows():
            print(f"Method: {method} | Instance: {row['instance_folder']}")
            print(f"  • Avg. makespan     : {row['avg_makespan']:.3f} (var {row['var_makespan']:.3f})")
            print(f"  • Avg. online time  : {row['avg_online_time']:.3f} (var {row['var_online_time']:.3f})")
            print(f"  • Avg. offline time : {row['avg_offline_time']:.3f} (var {row['var_offline_time']:.3f})")
            print(f"  • Feasibility ratio : {row['feasibility_ratio']:.3%}")
            print("-" * 50)

def summarize_feasibility(df):
    feasibility_summary = df.groupby(['method', 'instance_folder'])['feasibility'].agg(['count', 'sum'])
    feasibility_summary['ratio'] = feasibility_summary['sum'] / feasibility_summary['count']
    print("\n=== Feasibility Summary ===")
    print(feasibility_summary)

def wilcoxon_test(df):
    pivot_df = df.pivot_table(
        index='instance_folder',
        columns='method',
        values='obj',
        aggfunc='mean'  # or 'median' depending on your preference
    )

    methods = pivot_df.columns

    print("\n=== Wilcoxon Signed-Rank Test Results ===")
    for method1, method2 in combinations(methods, 2):
        x = pivot_df[method1]
        y = pivot_df[method2]

        # Drop any pair where either value is NaN or inf
        valid_idx = x.index[np.isfinite(x) & np.isfinite(y)]
        x_clean = x.loc[valid_idx]
        y_clean = y.loc[valid_idx]

        try:
            if len(x_clean) < 1:
                print(f"{method1} vs {method2} → Not enough valid samples.")
                continue
            stat, p = wilcoxon(x_clean, y_clean)
            print(f"{method1} vs {method2} → W={stat:.3f}, p={p:.4f}")
        except ValueError as e:
            print(f"{method1} vs {method2} → Error: {e}")



# evaluate_results("05_12_2025,12_39")