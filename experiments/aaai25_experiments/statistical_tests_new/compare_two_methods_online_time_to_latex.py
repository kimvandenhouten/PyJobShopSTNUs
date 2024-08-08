import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import ast
from general.latex_table_from_list import generate_latex_table_from_lists
import scipy
from scipy.stats import binomtest
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from experiments.aaai25_experiments.statistical_tests_new.proportion_test import proportion_test
from experiments.aaai25_experiments.statistical_tests_new.magnitude_test import magnitude_test

### START SETTINGS ###
noise_factor = 2
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
data.loc[data['feasibility'] == False, 'time_online'] = np.inf

data.to_csv(f'experiments/aaai25_experiments/final_results/combined_results_noise_factor={noise_factor}.csv')
methods = ["proactive_quantile_0.9", "STNU_robust", "reactive", "proactive_SAA_smart"]

if noise_factor == 1:
    method_pairs_default = [("proactive_quantile_0.9", "proactive_SAA_smart"),("proactive_quantile_0.9", "STNU_robust"),
                    ("proactive_SAA_smart", "STNU_robust"), ("proactive_quantile_0.9", "reactive"),
                    ("proactive_SAA_smart", "reactive"), ("STNU_robust", "reactive")]

    method_pairs_problems = {}
    for prob in ["j10", "j20", "j30", "ubo50", "ubo100"]:
        method_pairs_problems[prob] = method_pairs_default

    method_pairs = [("proactive_SAA_smart", "proactive_quantile_0.9"), ("proactive_quantile_0.9", "STNU_robust"),
                    ("proactive_SAA_smart", "STNU_robust"), ("proactive_quantile_0.9", "reactive"),
                    ("proactive_SAA_smart", "reactive"), ("STNU_robust", "reactive")]

    for prob in ["ubo50", "ubo100"]:
        method_pairs_problems[prob] = method_pairs
else:
    method_pairs_default = [("proactive_quantile_0.9", "proactive_SAA_smart"),
                            ("proactive_quantile_0.9", "STNU_robust"),
                            ("proactive_SAA_smart", "STNU_robust"), ("proactive_quantile_0.9", "reactive"),
                            ("proactive_SAA_smart", "reactive"), ("STNU_robust", "reactive")]

    method_pairs_problems = {}
    for prob in ["j10", "j20", "j30", "ubo50", "ubo100"]:
        method_pairs_problems[prob] = method_pairs_default

    method_pairs = [("proactive_SAA_smart", "proactive_quantile_0.9"), ("proactive_quantile_0.9", "STNU_robust"),
                    ("proactive_SAA_smart", "STNU_robust"), ("proactive_quantile_0.9", "reactive"),
                    ("proactive_SAA_smart", "reactive"), ("STNU_robust", "reactive")]

    for prob in ["j30"]:
        method_pairs_problems[prob] = method_pairs

    method_pairs = [("proactive_quantile_0.9", "proactive_SAA_smart"), ("STNU_robust", "proactive_quantile_0.9"),
                    ("STNU_robust", "proactive_SAA_smart"), ("proactive_quantile_0.9", "reactive"),
                    ("proactive_SAA_smart", "reactive"), ("STNU_robust", "reactive")]

    for prob in ["ubo50"]:
        method_pairs_problems[prob] = method_pairs

    method_pairs = [("proactive_SAA_smart", "proactive_quantile_0.9"), ("STNU_robust", "proactive_quantile_0.9"),
                    ( "STNU_robust", "proactive_SAA_smart",), ("proactive_quantile_0.9", "reactive"),
                    ("proactive_SAA_smart", "reactive"), ("STNU_robust", "reactive")]

    for prob in ["ubo100"]:
        method_pairs_problems[prob] = method_pairs

trans_dict = {"STNU_robust": "stnu",
              "reactive": "reactive",
              "proactive_quantile_0.9": "proactive$_{0.9}$",
               "proactive_SAA_smart": "proactive$_{SAA}$"}

# Define problem domains on which we will do the tests (from PSPlib instance sets)
problems = ["j10", "j20", "j30", "ubo50", "ubo100"]

# Significant levels for the different tests
alpha_consistent = 0.05
alpha_magnitude = 0.05
alpha_proportion = 0.05
### END OF SETTINGS ###

# Start preparing the data, substitute inf values
objectives = data["time_online"].tolist()
objectives = [i for i in objectives if i < np.inf]
inf_value = max(objectives) * 5
data.replace([np.inf], inf_value, inplace=True)

# Dictionary to store test results
test_results = {}
test_results_double_hits = {}
test_results_magnitude = {}
test_results_proportion = {}
# Loop over each problem domain

for problem in problems:
    test_results[problem] = {}
    test_results_magnitude[problem] = {}
    test_results_proportion[problem] = {}
    # Filter data for the current problem domain
    domain_data = data[data['instance_folder'] == problem]
    method_pairs = method_pairs_problems[problem]
    for (method1, method2) in method_pairs:
        ### START WILCOXON TEST ###
        test_results[problem][(method1, method2)] = {}

        # Filter data for both methods
        data1 = domain_data[domain_data['method'] == method1]
        data2 = domain_data[domain_data['method'] == method2]

        data1_list = data1["time_online"].tolist()
        data2_list = data2["time_online"].tolist()

        # Remove infeasible by both methods
        at_least_one_feasible_indices = []
        for i in range(len(data1_list)):
            if data1_list[i] < inf_value or data2_list[i] < inf_value:
                # print(f'Double hit for {data1_list[i]} and {data2_list[i]}')
                at_least_one_feasible_indices.append(i)

        data1_at_least_one_feasible = data1.iloc[at_least_one_feasible_indices]
        data1_at_least_one_feasible = data1_at_least_one_feasible.reset_index(drop=True)
        data2_at_least_one_feasible = data2.iloc[at_least_one_feasible_indices]
        data2_at_least_one_feasible = data2_at_least_one_feasible.reset_index(drop=True)
        data1_list = data1_at_least_one_feasible["time_online"].tolist()
        data2_list = data2_at_least_one_feasible['time_online'].tolist()

        # Compute differences that are needed for Wilcoxon test
        differences = np.array(data1_at_least_one_feasible['time_online'].tolist()) - np.array(data2_at_least_one_feasible['time_online'].tolist())


        # If all differences are zero we cannot do this test
        if sum(differences) == 0:
            test_results[problem][(method1, method2)] = {
                'obj': {'statistic': 9999, 'p-value': 9999, 'z-statistic': 9999,
                        "n_pairs": len(data1_at_least_one_feasible['time_online'].tolist()),
                        "sum_pos_ranks": 0, 'sum_neg_ranks': 0}}
        else:
            res = wilcoxon(data1_at_least_one_feasible['time_online'], data2_at_least_one_feasible['time_online'],
                           method="approx", zero_method="pratt")

            # Now also analyze which method is better by looking at the ranksums
            ranks = scipy.stats.rankdata([abs(x) for x in differences])
            signed_ranks = []
            for diff, rank in zip(differences, ranks):
                # Pratt method ignores zero ranks, although there were included in the ranking process
                if diff > 0:
                    signed_ranks.append(rank)
                elif diff < 0:
                    signed_ranks.append(-rank)

            # Sum of positive and negative ranks
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
                        "n_pairs": len(data1_at_least_one_feasible['time_online'].tolist()),
                        "sum_pos_ranks": sum_positive_ranks, 'sum_neg_ranks': sum_negative_ranks}
            }

            ### START PROPORTION TEST ###
            # For the proportion test we must count trials, wins and ties
            num_wins_1, num_trials, ties = 0, 0, 0
            for i in range(len(data1_list)):
                if data1_list[i] == data2_list[i]:
                    ties += 1
                else:
                    num_trials += 1
                    if data1_list[i] < data2_list[i]:
                        num_wins_1 += 1

            # We can only run the proportion test (binomtest) if not all trials were ties
            if num_trials > 0:
                z_value, p_value, null, sample_proportion = proportion_test(n=num_trials, k=num_wins_1, p_zero=0.5,
                                                                            z_crit=1.96)
                test_results_proportion[problem][(method1, method2)] = {
                    'obj': {'sample_proportion': sample_proportion, 'p-value': p_value, 'n_pairs': num_trials,
                            'ties': ties, 'z-statistic': z_value}}
            # If all trials were ties no test results exists
            else:
                test_results_proportion[problem][(method1, method2)] = {
                    'obj': {'sample_proportion': 9999, 'p-value': 9999,
                            'n_pairs': 9999, 'ties': ties,
                            'z-statistic': 9999}}

        ### START MAGNITUDE TEST ###
        # Now we will only use double hits.
            # Now we will only use double hits.
            data1_list = data1["time_online"].tolist()
            data2_list = data2["time_online"].tolist()

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
            data1_list = data1_double_hits['time_online'].tolist()
            data2_list = data2_double_hits['time_online'].tolist()

            # Run the test on the normalized data using the SciPy t-test
            result, mean_1, mean_2, n = magnitude_test(obs_1=data1_list, obs_2=data2_list)

            # We store the test results
            test_results_magnitude[problem][(method1, method2)] = {
                'obj': {'statistic': result.statistic, 'p-value': result.pvalue, "n_pairs": n,
                        "mean_method1": mean_1, "mean_method2": mean_2}}



### START PREPARING THE LATEX TABLES ###
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
                print(f'WARNING pair[1] is better {pair[1]}')
            else:
                better = pair[0]

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
            print(f"{pair}  There is a significant proportion of wins in objective {better} "
                  f"performs significantly better with proportion {result['sample_proportion']} and p-value "
                  f"{result['p-value']} and z-value {np.round(result['z-statistic'], 3)}.")
            overleaf_string = f"[{result['n_pairs']}] {np.round(result['sample_proportion'], 3)} (*)"
            print(f"Overleaf string: {overleaf_string}")
        else:
            print(f"{pair} There is no significant proportion of wins in obj: No significant difference."
                  f"proportion {result['sample_proportion']} and p-value {result['p-value']} and z-value {np.round(result['z-statistic'], 3)}")
            overleaf_string = f"[{result['n_pairs']}] {np.round(result['sample_proportion'], 3)} ({np.round(result['p-value'],3)})"
            print(f"Overleaf string: {overleaf_string}")
        new_row.append(overleaf_string)
    rows.append(new_row)

print(rows)
caption = (f"Pairwise comparison on time online for noise factor c={noise_factor}."
           f" Using a Wilcoxon test and a proportion test. Including all instances for which at least one"
           f" of the two methods found a feasible solution. Note that the ordering matters: the first"
           f" method showed is the better of the two in each paier method 1 - method 2. Each cell shows"
           f" on the first row [nr pairs] z-value (p-value) of the Wilcoxon test with (*) for p  $<$ 0.05."
           f" Each cell shows on the second row [nr pairs] proportion (p-value) with (*) for p $<$ 0.05 ")

latex_code_wilcoxon = generate_latex_table_from_lists(rows, caption=caption, label=f"tab:online_pairwise_{noise_factor}")


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
            else:
                better = pair[1]
                print(f'warning the second is better {pair[1]}')
            print(f"Double hits magnitude: {better} performs significantly better with"
                  f" stat {np.round(result['statistic'], 3)} and p-value {result['p-value']}.")
            overleaf_1 = f"[{result['n_pairs']}] {np.round(result['statistic'], 3)} (*) "
            overleaf_2 = f"{trans_dict[pair[0]]}: {np.round(result['mean_method1'], 2)}"
            overleaf_3 = f"{trans_dict[pair[1]]}: {np.round(result['mean_method2'], 2)}"
            print(overleaf_1)
            print(overleaf_2)
            print(overleaf_3)

        else:
            print(f"Double hits magnitude: No significant difference. With stat"
                  f" {np.round(result['statistic'], 3)} and p-value {result['p-value']}")
            overleaf_1 = f"[{result['n_pairs']}] {np.round(result['statistic'], 3)} ({np.round(result['p-value'],3)})"
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

caption = (f"Magnitude test on time online for noise factor c={noise_factor}."
           f" Using a pairwise t-test, including all instances for which both methods found a feasible solution,"
           f" and for which earlier tests indicated a significant consistent or proportional difference. Each cell "
           f" shows on the first row [nr pairs] t-stat (p-value) with (*) for p $<$ 0.05 and on the second row the normalized average of method 1"
           f" and on the third row the normalized average of method 2.")
latex_code_magnitude = generate_latex_table_from_lists(rows, caption=caption, label=f"tab:online_magnitude_{noise_factor}")
print(latex_code_wilcoxon)
print(latex_code_magnitude)
