from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark

DIRECTORY_INSTANCES = 'rcpsp_max/data'
INSTANCE_FOLDERS = ["ubo50"]
INSTANCE_IDS = [4]
noise_factor = 1

for instance_folder in INSTANCE_FOLDERS:
    for instance_id in INSTANCE_IDS:
        rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor)

        sampled_durations = rcpsp_max.sample_durations(1)

        for sample in sampled_durations:
            res, data_df = rcpsp_max.solve_with_warmstart(sample, time_limit=20, mode="NotQuiet")
            initial_solution = {}
            for index, row in data_df.iterrows():
                initial_solution[f'T{index}'] = {'start': row['start'], 'end': row['end']}
            print(initial_solution)

            res, data_df = rcpsp_max.solve_with_warmstart(sample, initial_solution=initial_solution, time_limit=5, mode="NotQuiet")

