import matplotlib.pyplot as plt
import os

def plot_simulation_statistics(makespans, violations, total_runs, output_folder="PyJobShopIntegration/images"):
    os.makedirs(output_folder, exist_ok=True)

    plt.figure()
    plt.bar(["Met", "Violated"], [total_runs - violations, violations])
    plt.title("Deadline Compliance")
    plt.ylabel("Number of Runs")
    plt.savefig(os.path.join(output_folder, "deadline_violations.png"))
    plt.show()

    plt.figure()
    plt.hist(makespans, bins=15)
    plt.title("Makespan Distribution")
    plt.xlabel("Makespan")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_folder, "makespan_distribution.png"))
    plt.show()
