import pandas as pd

noise_factor = 2
# Please refer to the csv file including all results from the experiments
if noise_factor == 1:
    data_1 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_1_07_08_2024,09_35.csv')
    data_2 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_1_07_10_2024,10_17.csv')

else:
    data_1 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_2_07_09_2024,07_10.csv')
    data_2 = pd.read_csv(f'experiments/aaai25_experiments/final_results/final_results_2_07_10_2024,10_17.csv')


data = pd.concat([data_1, data_2])

aggregation = data.groupby(['method', 'instance_folder', 'feasibility']).size().unstack(fill_value=0)

# Rename columns for clarity
aggregation.columns = ['False_Count', 'True_Count']

# Calculate total and True/Total ratio
aggregation['Total'] = aggregation['False_Count'] + aggregation['True_Count']
aggregation['True/Total'] = aggregation['True_Count'] / aggregation['Total']

print(aggregation)

pivot_table = aggregation['True/Total'].unstack('method')
# Rename columns to LaTeX format
new_column_names = {
    'STNU_robust': 'stnu',
    'proactive_SAA_smart': "proactive_SAA",
    'proactive_quantile_0.9': 'proactive_0.9',
    'reactive': "reactive"
    # Add other method mappings here if needed
}
pivot_table.columns = [new_column_names.get(col, col) for col in pivot_table.columns]

# Reindex to change row order
new_order = ['j10', 'j20', 'j30', 'ubo50', 'ubo100']
pivot_table = pivot_table.reindex(new_order)
# Generate LaTeX table
latex_table = pivot_table.to_latex(float_format="%.2f")
print(latex_table)
