import os
import matplotlib.pyplot as plt
from pyjobshop.plot import plot_task_gantt, plot_machine_gantt, plot_resource_usage

def plot_simulation_gantt(simulated_solution, model, filename=None, plot_type="auto", output_folder="PyJobShopIntegration/images"):
    """
    Plot a Gantt chart based on the model type and simulation result.

    :param simulated_solution: The Solution object from simulation
    :param model: The PyJobShop model (contains task/resource info)
    :param filename: Where to save the plot (if None, display only)
    :param plot_type: "auto", "machine", or "task"
    :param output_folder: Folder to save plots to
    """
    data = model.data()

    if plot_type == "auto":
        if data.num_machines > 0:
            plot_type = "machine"
        else:
            plot_type = "task"

    if plot_type == "machine":
        plt.figure(figsize=(12, 6))
        plot_machine_gantt(simulated_solution, data, plot_labels=True)
    elif plot_type == "task":
        fig, axes = plt.subplots(data.num_resources + 1, figsize=(12, 16),
                                 gridspec_kw={"height_ratios": [6] + [1] * data.num_resources})
        plot_task_gantt(simulated_solution, data, ax=axes[0])
        plot_resource_usage(simulated_solution, data, axes=axes[1:])
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    if filename:
        os.makedirs(output_folder, exist_ok=True)
        full_path = os.path.join(output_folder, filename)
        plt.savefig(full_path)
        print(f"[INFO] Saved Gantt plot to {full_path}")
    plt.show()
