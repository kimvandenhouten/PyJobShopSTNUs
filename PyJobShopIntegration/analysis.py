import os.path
import pandas as pd
folder = os.path.join("results", "mmrcpspd_used", "db")
desired_mode_order = ["robust", "0.9", "0.75", "0.5"]
metrics = {}
feasibility = {}
for file in os.listdir(folder):
    if file.startswith("evaluation") and file.endswith(".csv"):
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        mode = df['method'][0].split('_')[-1]
        for index, row in df.iterrows():
            instance = row['instance']
            method = row['method'].split('_')[0]
            noise = row['noise']
            if method not in metrics:
                metrics[method] = {}
            if instance not in metrics[method]:
                metrics[method][instance] = {}
            if noise not in metrics[method][instance]:
                metrics[method][instance][noise] = {
                    'avg_makespan': [0 for _ in desired_mode_order],
                    'avg_online_time': [0 for _ in desired_mode_order],
                    'avg_offline_time': [0 for _ in desired_mode_order]
                }
            makespan = row['avg_makespan']
            online_time = row['avg_online_time']
            offline_time = row['avg_offline_time']
            metrics[method][instance][noise]['avg_makespan'][desired_mode_order.index(mode)] = makespan
            metrics[method][instance][noise]['avg_online_time'][desired_mode_order.index(mode)] = online_time
            metrics[method][instance][noise]['avg_offline_time'][desired_mode_order.index(mode)] = offline_time
    elif file.startswith("feasibility") and file.endswith(".csv"):
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        mode = df['method'][0].split('_')[-1]
        for index, row in df.iterrows():
            instance = row['instance_folder']
            method = row['method'].split('_')[0]
            try:
                noise = int(row['noise_factor'])
            except ValueError:
                noise = row['noise_factor']
            if method not in feasibility:
                feasibility[method] = {}
            if instance not in feasibility[method]:
                feasibility[method][instance] = {}
            if noise not in feasibility[method][instance]:
                feasibility[method][instance][noise] = [0 for _ in desired_mode_order]
            feas = row['ratio']
            feasibility[method][instance][noise][desired_mode_order.index(mode)] = feas
print("Metrics:", metrics, "\nFeasibility:", feasibility)