from collections import defaultdict

from matplotlib import pyplot as plt
from pandas import DataFrame

from PyJobShopIntegration.utils import get_project_root
import pandas as pd
from general.logger import get_logger
from scipy.stats import wilcoxon, rankdata
from itertools import combinations
import numpy as np

logger = get_logger(__name__)

def evaluate_results(now):
    file = get_project_root() / "PyJobShopIntegration" / "results" / f"final_results_{now}.csv"
    report_file = get_project_root() / "PyJobShopIntegration" / "results" / f"evaluation_report_{now}.txt"

    logger.info(f"Evaluating results from {file}")
    df = pd.read_csv(file)

    with open(report_file, "w") as output:
        df_stnu = df[df['method'] == 'stnu']
        df_proactive = df[df['method'].str.startswith('proactive')]
        df_reactive = df[df['method'] == 'reactive']

        evaluate_methods(df, output)
        summarize_feasibility(df, output)
        wilcoxon_test(df, output)

def evaluate_methods(df, output):
    for method in df['method'].unique():
        print(f"\n=== Evaluating method: {method} ===", file=output)
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
        # summary = DataFrame()
        summary = {}
        for instance_folder in method_df['instance_folder'].unique():
            instance_df = method_df[method_df['instance_folder'] == instance_folder]
            # summary = pd.concat([summary, instance_df])
            summary[instance_folder] = {}
            summary[instance_folder]['avg_makespan'] = instance_df['obj'].mean()
            summary[instance_folder]['var_makespan'] = instance_df['obj'].var()
            summary[instance_folder]['avg_online_time'] = instance_df['time_online'].mean()
            summary[instance_folder]['var_online_time'] = instance_df['time_online'].var()
            summary[instance_folder]['avg_offline_time'] = instance_df['time_offline'].mean()
            summary[instance_folder]['var_offline_time'] = instance_df['time_offline'].var()
            summary[instance_folder]['feasibility_ratio'] = instance_df['feasibility'].mean()

        print("\n=== Method Evaluation Summary ===\n", file=output)
        for instance_folder, metrics in summary.items():
            print(f"Instance: {instance_folder}", file=output)
            for metric, value in metrics.items():
                print(f"  • {metric.replace('_', ' ').title()} : {value:.5f}", file=output)
            print("-" * 50, file=output)

def summarize_feasibility(df, output):
    feasibility_summary = df.groupby(['method', 'instance_folder'])['feasibility'].agg(['count', 'sum'])
    feasibility_summary['ratio'] = feasibility_summary['sum'] / feasibility_summary['count']
    print("\n=== Feasibility Summary ===", file=output)
    print(feasibility_summary, file=output)


def _perform_wilcoxon(metric_df, methods, alpha, min_samples):
    metric_results = defaultdict(dict)
    pivot_df = metric_df.pivot(columns='method', values='value')

    for i in methods:
        for j in methods:
            if i == j:
                metric_results[i][j] = {'p': None, 'significant': False, 'better': None}
                continue

            scores_i = pivot_df[i].dropna().reset_index()
            scores_j = pivot_df[j].dropna().reset_index()

            if len(scores_i) < min_samples or len(scores_j) < min_samples:
                metric_results[i][j] = {'p': None, 'significant': False, 'better': None}
                continue

            aligned = pd.concat([scores_i, scores_j], axis=1, join="inner").dropna()
            if aligned.shape[0] < min_samples:
                metric_results[i][j] = {'p': None, 'significant': False, 'better': None}
                continue

            try:
                stat, p = wilcoxon(aligned[i], aligned[j])
                significant = p < alpha

                differences = np.array(aligned[i]) - np.array(aligned[j])
                ranks = rankdata([abs(diff) for diff in differences])
                signed_ranks = [rank if diff > 0 else -rank for diff, rank in zip(differences, ranks) if diff != 0]

                sum_pos = sum(rank for rank in signed_ranks if rank > 0)
                sum_neg = sum(-rank for rank in signed_ranks if rank < 0)

                better = i if sum_pos > sum_neg else (j if sum_neg > sum_pos else None)

                metric_results[i][j] = {
                    'p': p,
                    'significant': significant,
                    'better': better,
                    'sum_pos_ranks': sum_pos,
                    'sum_neg_ranks': sum_neg,
                    'n_pairs': len(aligned),
                }
            except ValueError:
                metric_results[i][j] = {'p': None, 'significant': False, 'better': None}

    return metric_results


def wilcoxon_test(df, output, alpha=0.05, min_samples=2):
    print("\n=== Wilcoxon Test Results ===", file=output)
    methods = df['method'].unique()

    for metric in ['obj', 'time_online', 'time_offline']:
        print(f"\n--- Metric: {metric} ---", file=output)

        metric_df = df[['method', metric]].copy()
        metric_df.columns = ['method', 'value']

        results = _perform_wilcoxon(metric_df, methods, alpha, min_samples)

        for i in methods:
            for j in methods:
                if results[i][j]['p'] is not None:
                    print(
                        f"{i} vs {j}: p-value = {results[i][j]['p']:.5f}, significant = {results[i][j]['significant']}, "
                        f"better = {results[i][j]['better']}, sum_pos_ranks = {results[i][j]['sum_pos_ranks']}, "
                        f"sum_neg_ranks = {results[i][j]['sum_neg_ranks']}, n_pairs = {results[i][j]['n_pairs']}",
                        file=output)
                else:
                    print(f"{i} vs {j}: Not enough data for comparison", file=output)

# Example usage:
evaluate_results("05_15_2025,17_33")
