import csv
from collections import defaultdict
from matplotlib import pyplot as plt
from pandas import DataFrame
from PyJobShopIntegration.utils import get_project_root
import pandas as pd
from general.logger import get_logger
from scipy.stats import wilcoxon, rankdata
import numpy as np
from itertools import combinations

logger = get_logger(__name__)


def evaluate_results(now):
    root = get_project_root() / "PyJobShopIntegration" / "results"
    file = root / f"final_results_{now}.csv"
    df = pd.read_csv(file)

    logger.info(f"Evaluating results from {file}")

    # Output files
    eval_csv = root / f"evaluation_summary_{now}.csv"
    feas_csv = root / f"feasibility_summary_{now}.csv"
    wilcox_csv = root / f"wilcoxon_results_{now}.csv"

    evaluate_methods(df, eval_csv)
    summarize_feasibility(df, feas_csv)
    wilcoxon_test(df, wilcox_csv)


def evaluate_methods(df, out_file):
    rows = []
    for method in df['method'].unique():
        method_df = df[df['method'] == method].copy()
        method_df['obj'] = pd.to_numeric(method_df['obj'], errors='coerce')
        method_df['time_online'] = pd.to_numeric(method_df['time_online'], errors='coerce')
        method_df['time_offline'] = pd.to_numeric(method_df['time_offline'], errors='coerce')
        method_df['feasibility'] = pd.to_numeric(method_df['feasibility'], errors='coerce')

        mask = (
                np.isfinite(method_df['obj']) &
                np.isfinite(method_df['time_online']) &
                np.isfinite(method_df['time_offline']) &
                np.isfinite(method_df['feasibility'])
        )
        method_df = method_df[mask]

        for instance_folder in method_df['instance_folder'].unique():
            folder_df = method_df[method_df['instance_folder'] == instance_folder]

            for noise in sorted(folder_df['noise_factor'].unique()):
                sub_df = folder_df[folder_df['noise_factor'] == noise]

                rows.append({
                    'method': method,
                    'instance': instance_folder,
                    'noise': noise,
                    'avg_makespan': sub_df['obj'].mean(),
                    'var_makespan': sub_df['obj'].var(),
                    'avg_online_time': sub_df['time_online'].mean(),
                    'var_online_time': sub_df['time_online'].var(),
                    'avg_offline_time': sub_df['time_offline'].mean(),
                    'var_offline_time': sub_df['time_offline'].var(),
                })

    pd.DataFrame(rows).to_csv(out_file, index=False)


def summarize_feasibility(df, out_file):
    # Summary by method and instance
    summary1 = df.groupby(['method', 'instance_folder'])['feasibility'].agg(['count', 'sum']).reset_index()
    summary1['ratio'] = summary1['sum'] / summary1['count']
    summary1['noise_factor'] = 'ALL'

    # Summary by method, instance, noise
    summary2 = df.groupby(['method', 'instance_folder', 'noise_factor'])['feasibility'].agg(['count', 'sum']).reset_index()
    summary2['ratio'] = summary2['sum'] / summary2['count']

    # Standardize column names
    summary1.columns = ['method', 'instance_folder', 'count', 'sum', 'ratio', 'noise_factor']
    summary2.columns = ['method', 'instance_folder', 'noise_factor', 'count', 'sum', 'ratio']

    # Combine and write to file
    combined = pd.concat([summary1, summary2], ignore_index=True)
    combined.to_csv(out_file, index=False)


def _perform_wilcoxon(metric_df, methods, alpha, min_samples):
    metric_results = []

    pivot_df = metric_df.pivot(columns='method', values='value')

    for i in methods:
        for j in methods:
            result = {
                'method_1': i,
                'method_2': j,
                'metric': metric_df.name,
                'p_value': None,
                'significant': False,
                'better': None,
                'sum_pos_ranks': None,
                'sum_neg_ranks': None,
                'n_pairs': 0,
            }

            if i == j:
                metric_results.append(result)
                continue

            scores_i = pivot_df[i].dropna().reset_index(drop=True)
            scores_j = pivot_df[j].dropna().reset_index(drop=True)

            if len(scores_i) < min_samples or len(scores_j) < min_samples:
                metric_results.append(result)
                continue

            aligned = pd.concat([scores_i, scores_j], axis=1, join="inner").dropna()
            aligned = aligned[np.isfinite(aligned).all(axis=1)]
            if aligned.shape[0] < min_samples:
                metric_results.append(result)
                continue

            try:
                stat, p = wilcoxon(aligned.iloc[:, 0], aligned.iloc[:, 1])
                differences = np.array(aligned.iloc[:, 1]) - np.array(aligned.iloc[:, 0])
                ranks = rankdata(np.abs(differences))
                signed_ranks = [rank if diff > 0 else -rank for diff, rank in zip(differences, ranks) if diff != 0]

                sum_pos = sum(rank for rank in signed_ranks if rank > 0)
                sum_neg = sum(-rank for rank in signed_ranks if rank < 0)
                better = i if sum_pos > sum_neg else (j if sum_neg > sum_pos else "Equal")

                result.update({
                    'p_value': p,
                    'significant': p < alpha,
                    'better': better,
                    'sum_pos_ranks': sum_pos,
                    'sum_neg_ranks': sum_neg,
                    'n_pairs': len(aligned),
                })
            except ValueError:
                pass

            metric_results.append(result)

    return metric_results


def wilcoxon_test(df, out_file, alpha=0.05, min_samples=2):
    all_results = []

    for metric in ['obj', 'time_online', 'time_offline']:
        metric_df = df[['method', metric]].copy()
        metric_df.columns = ['method', 'value']
        metric_df.name = metric

        methods = df['method'].unique()
        results = _perform_wilcoxon(metric_df, methods, alpha, min_samples)
        all_results.extend(results)

    pd.DataFrame(all_results).to_csv(out_file, index=False)


# Example usage:
evaluate_results("05_26_2025,00_17")
evaluate_results("05_26_2025,10_46")
evaluate_results("05_26_2025,11_53")
evaluate_results("05_26_2025,13_04")
evaluate_results("05_26_2025,15_05")
