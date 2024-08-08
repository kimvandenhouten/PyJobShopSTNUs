import pandas as pd
import itertools
import numpy as np
from general.latex_table_from_list import generate_latex_table_from_lists
from scipy.stats import binomtest
from scipy.stats import wilcoxon
import scipy
from experiments.aaai25_experiments.statistical_tests.proportion_test import proportion_test
from experiments.aaai25_experiments.statistical_tests.magnitude_test import magnitude_test

"""
This script contains the statistical tests that generate the tables that are presented in the Technical Appendix
of the AAAI25 submission "Proactive and Reactive Constraint Programming for Stochastic Project Scheduling with Maximal
Time-Lags" including the Wilcoxon, proportion test, and magnitude test for offline runtime.
"""

### SETTINGS ###
noise_factor = 1
# Please refer to the csv file including all results from the experiments
if noise_factor == 1:
    data_1 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_1_07_08_2024,09_35.csv')
    data_2 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_1_07_10_2024,10_17.csv')
    data = pd.concat([data_1, data_2])
else:
    data_1 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_2_07_09_2024,07_10.csv')
    data_2 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_2_07_10_2024,10_17.csv')
    data_3 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_2_07_11_2024,10_51.csv')
    data = pd.concat([data_1, data_2, data_3])

# Correct infeasibilities
data.loc[data['feasibility'] == False, 'time_offline'] = np.inf

# DEFINE PAIRS OF METHOD THAT MUST BE COMPARED
# Note that these pairs are used to determine the ordering in the results table that is printed for overleaf
# List of all methods
methods = ["proactive_quantile_0.9", "STNU_robust", "reactive", "proactive_SAA_smart"]
if noise_factor == 1:
    method_pairs_default = [("proactive_quantile_0.9", "reactive"), ("reactive", "STNU_robust"),
                    ("reactive", "proactive_SAA_smart"), ("proactive_quantile_0.9", "STNU_robust"),
                    ("proactive_quantile_0.9", "proactive_SAA_smart"), ("proactive_SAA_smart", "STNU_robust")]

    method_pairs_problems = {}
    for prob in ["j10", "j20"]:
        method_pairs_problems[prob] = method_pairs_default

    for prob in ["j30", "ubo50", "ubo100"]:
        method_pairs_problems[prob] = [("proactive_quantile_0.9", "reactive"), ("reactive", "STNU_robust"),
                    ("reactive", "proactive_SAA_smart"), ("proactive_quantile_0.9", "STNU_robust"),
                    ("proactive_quantile_0.9", "proactive_SAA_smart"), ("STNU_robust", "proactive_SAA_smart")]


elif noise_factor == 2:
    method_pairs_default = [("proactive_quantile_0.9", "reactive"), ("reactive", "STNU_robust"),
                    ("reactive", "proactive_SAA_smart"), ("proactive_quantile_0.9", "STNU_robust"),
                    ("proactive_quantile_0.9", "proactive_SAA_smart"), ("proactive_SAA_smart", "STNU_robust")]

    method_pairs_problems = {}
    for prob in ["j10"]:
        method_pairs_problems[prob] = method_pairs_default

    for prob in ["j20", "j30"]:
        method_pairs_problems[prob] = [("proactive_quantile_0.9", "reactive"), ("reactive", "STNU_robust"),
                    ("reactive", "proactive_SAA_smart"), ("proactive_quantile_0.9", "STNU_robust"),
                    ("proactive_quantile_0.9", "proactive_SAA_smart"), ("STNU_robust", "proactive_SAA_smart")]

    for prob in ["ubo50", "ubo100"]:
        method_pairs_problems[prob] = [("STNU_robust", "proactive_quantile_0.9"), ("STNU_robust", "reactive"),
                                       ("STNU_robust", "proactive_SAA_smart"),
                                       ("proactive_quantile_0.9", "reactive"),
                                       ("proactive_quantile_0.9", "proactive_SAA_smart"),
                                       ("reactive", "proactive_SAA_smart")]


# This translation dict is used to translate the methods as described in the csv file to shorter version for overleaf
trans_dict = {"STNU_robust": "stnu",
              "reactive": "reactive",
              "proactive_quantile_0.9": "proactive$_{0.9}$",
               "proactive_SAA_smart": "proactive$_{SAA}$"}

# Define problem domains (instance sets PSPlib)
if noise_factor == 1:
    problems = ["j10", "j20", "j30", "ubo50", "ubo100"]
elif noise_factor == 2:
    problems = ["j10", "j20", "j30", "ubo50", "ubo100"]


# Significance levels for the different tests
alpha_consistent = 0.05
alpha_magnitude = 0.05
alpha_proportion = 0.05
### END OF SETTINGS ###

# Prepare the data, substitute inf values with a high enough value
objectives = data["time_offline"].tolist()
objectives = [i for i in objectives if i < np.inf]
inf_value = round(max(objectives) * 5)
#print(inf_value)
data.replace([np.inf], inf_value, inplace=True)

# Dictionary to store test results
test_results = {}
test_results_magnitude = {}
test_results_proportion = {}

# Loop over each problem domain
for problem in problems:
    # Make dicts for test results
    test_results[problem] = {}
    test_results_magnitude[problem] = {}
    test_results_proportion[problem] = {}

    # Filter data for the current problem domain
    domain_data = data[data['instance_folder'] == problem]
    method_pairs = method_pairs_problems[problem]
    for (method1, method2) in method_pairs:
        test_results[problem][(method1, method2)] = {}

        # Filter data for both methods
        data1 = domain_data[domain_data['method'] == method1]
        data2 = domain_data[domain_data['method'] == method2]

        data1_list = data1["time_offline"].tolist()
        data2_list = data2["time_offline"].tolist()

        # Remove instances that were not solved by both methods
        at_least_one_feasible_indices = []
        for i in range(len(data1_list)):
            if data1_list[i] < inf_value or data2_list[i] < inf_value:
                at_least_one_feasible_indices.append(i)
        data1_at_least_one_feasible = data1.iloc[at_least_one_feasible_indices]
        data1_at_least_one_feasible = data1_at_least_one_feasible.reset_index(drop=True)
        data2_at_least_one_feasible = data2.iloc[at_least_one_feasible_indices]
        data2_at_least_one_feasible = data2_at_least_one_feasible.reset_index(drop=True)

        # Prepare data for Wilcoxon test
        data1_list = data1_at_least_one_feasible["time_offline"].tolist()
        data2_list = data2_at_least_one_feasible['time_offline'].tolist()
        differences = np.array(data1_at_least_one_feasible['time_offline'].tolist()) - np.array(data2_at_least_one_feasible['time_offline'].tolist())

        # Note that we cannot run this test if all differences are zero
        if sum(differences) == 0:
            test_results[problem][(method1, method2)] = {
                'obj': {'statistic': np.nan, 'p-value': np.nan, 'z-statistic': np.nan,
                        "n_pairs": len(data1_at_least_one_feasible['time_offline'].tolist()),
                        "sum_pos_ranks": 0, 'sum_neg_ranks': 0}}
        else:
            # Run the test using the SciPy implementation, with the normal approximation (method="approx"), and the pratt
            # method for handling zero differences (zero_method="pratt"), and a correction for continuity (correction=True)
            res = wilcoxon(data1_at_least_one_feasible['time_offline'], data2_at_least_one_feasible['time_offline'],
                           method="approx", zero_method="pratt", correction=True)

            # Analyze the ranks to find which of the methods is outperforming
            ranks = scipy.stats.rankdata([abs(x) for x in differences])

            signed_ranks = []
            for diff, rank in zip(differences, ranks):
                # Pratt method ignores zero ranks, although there were included in the ranking process
                if diff > 0:
                    signed_ranks.append(rank)
                elif diff < 0:
                    signed_ranks.append(-rank)
            # Sum of positive and negative ranks which is needed to
            sum_positive_ranks = sum(rank for rank in signed_ranks if rank > 0)
            sum_negative_ranks = sum(-rank for rank in signed_ranks if rank < 0)

            if sum_positive_ranks > sum_negative_ranks:
                print(f'{method2} is probably better')
            else:
                print(f'{method1} is probably better')

            # Obtain the test statistics
            stat_obj = res.statistic
            p_obj = res.pvalue
            z_obj = res.zstatistic

            # Store results
            test_results[problem][(method1, method2)] = {
                'obj': {'statistic': stat_obj, 'p-value': p_obj, 'z-statistic': z_obj,
                        "n_pairs": len(data1_at_least_one_feasible['time_offline'].tolist()),
                        "sum_pos_ranks": sum_positive_ranks, 'sum_neg_ranks': sum_negative_ranks}
            }

        ### START PROPORTION TEST ###
        num_wins_1, num_trials, ties = 0, 0, 0
        for i in range(len(data1_list)):
            if data1_list[i] == data2_list[i]:
                ties += 1
            else:
                num_trials += 1
                if data1_list[i] < data2_list[i]:
                    num_wins_1 += 1
        print(
            f'total number of wins for {method1} is {num_wins_1} while for {method2} is {num_trials - num_wins_1}')

        # We can only run this test if not all pairs are ties
        if num_trials > 1:
            z_value, p_value, null, sample_proportion = proportion_test(n=num_trials, k=num_wins_1, p_zero=0.5,
                                                                        z_crit=1.96)
            test_results_proportion[problem][(method1, method2)] = {
                'obj': {'sample_proportion': sample_proportion, 'p-value': p_value, 'n_pairs': num_trials,
                        'ties': ties, 'z-statistic': z_value}}
        else:
            test_results_proportion[problem][(method1, method2)] = {
                'obj': {'sample_proportion': np.nan, 'p-value': np.nan,
                        'n_pairs': np.nan, 'ties': np.nan,
                        'z-statistic': np.nan}}

        ### START MAGNITUDE TEST ###
        # Now we will only use double hits.
        data1_list = data1["time_offline"].tolist()
        data2_list = data2["time_offline"].tolist()

        double_hits_indices = []
        for i in range(len(data1_list)):
            if data1_list[i] < inf_value and data2_list[i] < inf_value:
                double_hits_indices.append(i)

        # Select only double hits
        data1_double_hits = data1.iloc[double_hits_indices]
        data1_double_hits = data1_double_hits.reset_index(drop=True)
        data2_double_hits = data2.iloc[double_hits_indices]
        data2_double_hits = data2_double_hits.reset_index(drop=True)

        # Now do the magnitude test on the double hits
        data1_list = data1_double_hits['time_offline'].tolist()
        data2_list = data2_double_hits['time_offline'].tolist()

        # Run the test on the normalized data using the SciPy t-test
        result, mean_1, mean_2, n = magnitude_test(obs_1=data1_list, obs_2=data2_list)

        # We store the test results
        test_results_magnitude[problem][(method1, method2)] = {
        'obj': {'statistic': result.statistic, 'p-value': result.pvalue, "n_pairs": n,
                "mean_method1": mean_1, "mean_method2": mean_2}}



### START MAKING THE OVERLEAF TABLES ###
rows = []
for problem in problems:
    print(f'\nStart evaluation for new problem set {problem}')

    # Obtain Wilcoxon-stats and make overleaf cell
    method_pairs = method_pairs_problems[problem]
    method_pairs_to_header = [f"{trans_dict[pair[0]]}-{trans_dict[pair[1]]}" for pair in method_pairs]
    header = [problem] + method_pairs_to_header
    rows.append(header)

    new_row = [""]
    for pair in method_pairs:
        result = test_results[problem][(pair[0], pair[1])]['obj']

        if result['p-value'] < alpha_consistent:
            if result['sum_pos_ranks'] > result['sum_neg_ranks']:
                better = pair[1]
                worser = pair[0]
                print(f'WARNING pair[1] is better {pair[1]}')
            else:
                better = pair[0]
                worser = pair[1]

            print(
                f"Wilcoxon objective {pair}: {better} performs significantly better with z-stat {np.round(result['z-statistic'], 3)} and p-value "
                f"{result['p-value']}")
            overleaf_string = f"[{result['n_pairs']}] {np.round(result['z-statistic'], 3)} (*)"
            print(f"Overleaf string: {overleaf_string}")
        else:
            print(
                f"Wilcoxon objective {pair}: No significant difference with z-stat {np.round(result['z-statistic'], 3)} and p-value "
                f"{result['p-value']}.")
            overleaf_string = f"[{result['n_pairs']}] {np.round(result['z-statistic'], 3)} ({np.round(result['p-value'], 3)})"
            print(f"Overleaf string: {overleaf_string}")
        new_row.append(overleaf_string)
    rows.append(new_row)

    new_row = [""]
    for pair in method_pairs:
        result = test_results_proportion[problem][(pair[0], pair[1])]['obj']

        if result['p-value'] < alpha_proportion:
            if result['sample_proportion'] >= 0.5:
                better = pair[0]
            else:
                better = pair[1]
                print(f'WARNING pair[1] is better {pair[1]}')
            print(f"{pair} There is a significant proportion of wins in objective {better} "
                  f"performs significantly better with proportion {result['sample_proportion']} and p-value "
                  f"{result['p-value']} and z-value {np.round(result['z-statistic'], 3)}.")
            overleaf_string = f"[{result['n_pairs']}] {np.round(result['sample_proportion'], 3)} (*)"
            print(f"Overleaf string: {overleaf_string}")
        else:
            print(f"{pair}  There is no significant proportion of wins in obj: No significant difference."
                  f"proportion {result['sample_proportion']} and p-value {result['p-value']} and z-value "
                  f"{np.round(result['z-statistic'], 3)}")
            overleaf_string = (f"[{result['n_pairs']}] {np.round(result['sample_proportion'], 3)} "
                               f"({np.round(result['p-value'],3)})")
            print(f"Overleaf string: {overleaf_string}")
        new_row.append(overleaf_string)
    rows.append(new_row)

print(rows)
caption = (f"Pairwise comparison on time offline for noise factor c={noise_factor}."
           f" Using a Wilcoxon test and a proportion test. Including all instances for which at least one"
           f" of the two methods found a feasible solution. Note that the ordering matters: the first"
           f" method showed is the better of the two in each pair method 1 - method 2 according to Wilcoxon."
           f" Each cell shows"
           f" on the first row [nr pairs] z-value (p-value) of the Wilcoxon test with (*) for p  $<$ 0.05."
           f" Each cell shows on the second row [nr pairs] ratio of wins (p-value) with (*) for p $<$ 0.05 ")
latex_code_wilcoxon = generate_latex_table_from_lists(rows, caption=caption, label=f"tab:offline_pairwise_{noise_factor}")

rows = []

for problem in problems:
    method_pairs = method_pairs_problems[problem]
    method_pairs_to_header = [f"{trans_dict[pair[0]]}-{trans_dict[pair[1]]}" for pair in method_pairs]
    header = [problem] + method_pairs_to_header
    rows.append(header)
    print(f'\nStart evaluation for new problem set {problem}')
    new_row_1 = [""]
    new_row_2 = [""]
    new_row_3 = [""]
    # Obtain Wilcoxon-stats and make overleaf cell
    for pair in method_pairs:
        result = test_results_magnitude[problem][(pair[0], pair[1])]['obj']
        if result['p-value'] < alpha_magnitude:
            if result['statistic'] < 0:
                better = pair[0]
                worser = pair[1]
            else:
                better = pair[1]
                worser = pair[0]
                print(f'warning the second is better {pair[1]}')
            print(f"Double hits magnitude: {better} performs significantly better than {worser} with"
                  f" stat {np.round(result['statistic'], 3)} and p-value {result['p-value']}.")
            overleaf_1 = f"[{result['n_pairs']}] {np.round(result['statistic'], 3)} (*) "
            overleaf_2 = f"{trans_dict[pair[0]]}: {np.round(result['mean_method1'], 2)}"
            overleaf_3 = f"{trans_dict[pair[1]]}: {np.round(result['mean_method2'], 2)}"
            print(overleaf_1)
            print(overleaf_2)
            print(overleaf_3)

        else:
            print(f"Double hits magnitude: No significant difference between {pair}. With stat"
                  f" {np.round(result['statistic'], 3)} and p-value {result['p-value']}")
            overleaf_1 = f"[{result['n_pairs']}] {np.round(result['statistic'], 3)} ({np.round(result['p-value'], 3)})"
            overleaf_2 = f"{trans_dict[pair[0]]}: {np.round(result['mean_method1'], 2)}"
            overleaf_3 = f"{trans_dict[pair[1]]}: {np.round(result['mean_method2'], 2)}"
            print(overleaf_1)
            print(overleaf_2)
            print(overleaf_3)
        new_row_1.append(overleaf_1)
        new_row_2.append(overleaf_2)
        new_row_3.append(overleaf_3)
    rows.append(new_row_1)
    rows.append(new_row_2)
    rows.append(new_row_3)

caption = (f"Magnitude test on time offline for noise factor c={noise_factor}."
           f" Using a pairwise t-test, including all instances for which both methods found a feasible solution,"
           f" and for which earlier tests indicated a significant consistent or proportional difference. Each cell "
           f" shows on the first row [nr pairs] t-stat (p-value) with (*) for p $<$ 0.05 and on the second row the normalized average of method 1"
           f" and on the third row the normalized average of method 2.")
latex_code_magnitude = generate_latex_table_from_lists(rows, caption=caption, label=f"tab:offline_magnitude_{noise_factor}")
print(latex_code_wilcoxon)
print(latex_code_magnitude)