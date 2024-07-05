import pandas as pd

# Read proactive results
data_proactive = pd.read_csv("experiments/aaai25_experiments/results/tuning/proactive/results_proactive_tuning_proactive_quantile.csv")

# Read reactive results
data_reactive_j10 = pd.read_csv("experiments/aaai25_experiments/results/tuning/reactive/results_tuning_reactive_j10.csv")
data_reactive_ubo50 = pd.read_csv("experiments/aaai25_experiments/results/tuning/reactive/results_tuning_reactive_ubo50.csv")

# Combine
data = pd.concat([data_proactive, data_reactive_j10, data_reactive_ubo50])

# Aggregate
print(data)
aggregated_df = data.groupby(['instance_folder', 'instance_id', 'method', 'real_durations']).agg(
    obj =('obj', 'mean'),
).reset_index()

aggregated_df.to_csv(f'experiments/aaai25_experiments/results/tuning/proactive_reactive.csv', index=False)