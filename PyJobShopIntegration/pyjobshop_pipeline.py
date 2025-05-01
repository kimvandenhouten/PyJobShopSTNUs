import sys
import os
import datetime
from parser import parse_data
from evaluator import evaluate_results
problem_type = sys.argv[-1]
folder = problem_type
# SETTINGS HEURISTIC PROACTIVE APPROACH
mode_proactive = "quantile_0.9"
time_limit_proactive = 600
# SETTINGS REACTIVE APPROACH
time_limit_rescheduling = 2
# SETTINGS SAA APPROACH
mode_saa = "SAA_smart"
time_limit_saa = 1800
nb_scenarios_saa = 4
# SETTINGS STNU APPROACH
time_limit_cp_stnu = 600
mode_stnu = "robust"

# SETTINGS EXPERIMENTS
DIRECTORY_PI = os.path.join("aaai25", "results_perfect_information")
INSTANCE_FOLDERS = ["j10", "j20", "j30", "ubo50", "ubo100"]
INSTANCE_IDS = range(1, 51)
NOISE_FACTORS = [1, 2]
nb_scenarios_test = 10
proactive_reactive = True
proactive_saa = True
stnu = True
writing = False
now = datetime.datetime.now().strftime("%m_%d_%Y,%H_%M")
for noise_factor in NOISE_FACTORS:
    output_file = f'final_results_{noise_factor}_{now}.csv'
    data = []
    for instance_folder in INSTANCE_FOLDERS:
        folder_path = os.path.join("data", folder, instance_folder)
        for file in os.listdir(folder_path):
            # Load data
            instance = parse_data(os.path.join(folder_path, file), problem_type)
            test_durations_samples = instance.sample_durations(nb_scenarios_test)
            # Run experiments on proactive, reactive and stnu
            # TODO implement the proactive, reactive and stnu approaches possibly reusing already existing code
            for i, duration_sample in enumerate(test_durations_samples):
                if proactive_reactive:
                    pass
                if proactive_saa:
                    pass
                if stnu:
                    model = instance.create_model()
                    pass
    # Analyze the results perform statistical tests and create plots
    evaluate_results(data)

