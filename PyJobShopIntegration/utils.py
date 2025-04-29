import copy
from typing import Dict

import numpy as np
from pyjobshop import Solution
from temporal_networks.stnu import STNU
from temporal_networks.rte_star import RTEdata
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
def get_project_root()->Path:
    """Returns the root path of the project."""
    return Path(__file__).resolve().parents[1]

def find_schedule_per_resource(solution: Solution) -> Dict[int, list[int]]:
    """
    Returns a dictionary mapping resource indices to schedule of task indices.
    This function is authored by Joost Berkhout (comes from a private repository)
    Warning: if tasks start times are equal, the order is not guaranteed.
    """

    schedule_per_resource: dict[int, list[int]] = {}

    # Group tasks by resource
    for task_idx, task in enumerate(solution.tasks):
        for resource_idx in task.resources:
            if resource_idx not in schedule_per_resource:
                schedule_per_resource[resource_idx] = []
            schedule_per_resource[resource_idx].append(task_idx)

    # Sort tasks per resource by start time
    for resource_idx, task_indices in schedule_per_resource.items():
        schedule_per_resource[resource_idx] = sorted(
            task_indices, key=lambda idx: solution.tasks[idx].start
        )

    return schedule_per_resource


def remove_all_duplicates(tuples_list):
    """
    This is a helper function to remove unneeded duplicates in resource chains
    """
    unique_tuples = []
    seen = set()

    for current_tuple in tuples_list:
        if current_tuple not in seen:
            unique_tuples.append(current_tuple)
            seen.add(current_tuple)

    return unique_tuples


def get_resource_chains(schedule, capacity, resources, complete=False) -> (list[(int, int)], list[dict]):
    """
    This function implements the greedy interval scheduling algorithm from the book Algorithm Design by Jon Kleinberg
    and Eva Tardos, it assigns the tasks to sub-resources based on the start times. This function is relevant for RCPSP
    problems, as the solution of a CP model only provides start times, and for the construction of an STNU we need to
    add resource chains per subresource.
    """
    # schedule is a list of dicts of this form:
    # {"task": i, " "start": start, "end": end}
    reserved_until = {}
    for resource_index, resource_capacity in enumerate(capacity):
        reserved_until |= {resource_index: [0] * resource_capacity}

    resource_use = {}

    resource_assignment = []
    print(schedule)
    for d in sorted(schedule, key=lambda d: d['start']):
        for resource_index, required in enumerate(resources[d['task']]):
            reservations = reserved_until[resource_index]
            assigned = []
            for idx in range(len(reservations)):
                if len(assigned) == required:
                    break
                if reservations[idx] <= d['start']:
                    reservations[idx] = d['end']
                    assigned.append({'task': d['task'],
                                     'resource_group': resource_index,
                                     'id': idx})
                    users = resource_use.setdefault((resource_index, idx), [])
                    users.append(
                        {'Task': d['task'], 'Start': d['start']})

            if len(assigned) < required:
                ValueError(f'ERROR: only found {len(assigned)} of {required} resources (type {resource_index}) '
                      f'for task {d["task"]}')
            else:
                assert len(assigned) == required
                resource_assignment += assigned

    resource_chains = []
    if complete:
        for resource_activities in resource_use.values():
            if len(resource_activities) > 1:  # Check if there are multiple activities assigned to the same resource
                # Sort by start time
                resource_activities = sorted(resource_activities, key=lambda x: x["Start"])
                # To do keep track of edges that should be added to STN
                for i in range(1, len(resource_activities)):
                    for j in range(0, i):
                        predecessor = resource_activities[j]
                        successor = resource_activities[i]
                        resource_chains.append((predecessor["Task"],
                                                successor["Task"]))
    else:
        for resource_activities in resource_use.values():
            if len(resource_activities) > 1:  # Check if there are multiple activities assigned to the same resource
                # Sort by start time
                resource_activities = sorted(resource_activities, key=lambda x: x["Start"])

                # To do keep track of edges that should be added to STN
                for i in range(1, len(resource_activities)):
                    predecessor = resource_activities[i - 1]
                    successor = resource_activities[i]
                    resource_chains.append((predecessor["Task"],
                                            successor["Task"]))
    unique_tuples = remove_all_duplicates(resource_chains)
    return unique_tuples, resource_assignment


def add_resource_chains(stnu: STNU, resource_chains: list[(int, int)]):
    """
    This function adds the found resource chains to the STNU
    """
    for pred_task, succ_task in resource_chains:
        # the finish of the predecessor should precede the start of the successor
        pred_idx_finish = stnu.translation_dict_reversed[
            f"{pred_task}_{STNU.EVENT_FINISH}"]  # Get translation index from finish of predecessor
        suc_idx_start = stnu.translation_dict_reversed[
            f"{succ_task}_{STNU.EVENT_START}"]  # Get translation index from start of successor

        # add constraint between predecessor and successor
        stnu.set_ordinary_edge(suc_idx_start, pred_idx_finish, 0)

    return stnu


def get_start_and_finish_from_rte(estnu: STNU, rte_data:RTEdata, num_tasks: int) -> (list[int], list[int]):
    """
    This function can be used to link the start times and finish times from the rte_dta
    to the task indices
    """
    # TODO: can we make this faster / vectorize, or should it even be integrated in the RTE*?
    start_times, finish_times = [], []
    for task in range(num_tasks):
        node_idx_start = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_START}']
        node_idx_finish = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_FINISH}']
        start_times.append(rte_data.f[node_idx_start])
        finish_times.append(rte_data.f[node_idx_finish])
    return start_times, finish_times


def get_start_and_finish_from_rte(estnu: STNU, rte_data:RTEdata, num_tasks: int) -> (list[int], list[int]):
    """
    This function can be used to link the start times and finish times from the rte_dta
    to the task indices
    """
    # TODO: can we make this faster / vectorize, or should it even be integrated in the RTE*?
    start_times, finish_times = [], []
    for task in range(num_tasks):
        node_idx_start = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_START}']
        node_idx_finish = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_FINISH}']
        start_times.append(rte_data.f[node_idx_start])
        finish_times.append(rte_data.f[node_idx_finish])
    return start_times, finish_times

def overwrite_pyjobshop_solution(solution: Solution, start_times: list[int], finish_times: list[int])\
        -> (Solution, int):
    simulated_solution = copy.deepcopy(solution)

    for i in range(len(solution.tasks)):
        simulated_solution.tasks[i].start = start_times[i]
        simulated_solution.tasks[i].end = finish_times[i]

    return simulated_solution


def rte_data_to_pyjobshop_solution(solution: Solution, estnu: STNU, rte_data: RTEdata, num_tasks: int,
                                   objective: str="makespan") -> (Solution, int):
    """
    This function transforms the output of an RTE simulation into a PyJobShop solution
    """
    start_times, finish_times = get_start_and_finish_from_rte(estnu, rte_data, num_tasks)
    simulated_solution = overwrite_pyjobshop_solution(solution, start_times, finish_times)

    # TODO: can we make this automatically aligned with the PyJobShop model? Compute the objective given new
    #  start and finish times
    if objective == "makespan":
        objective_value = max(rte_data.f.values())
    else:
        raise NotImplementedError(f"Objective {objective} not")

    return simulated_solution, objective_value


def sample_for_rte(sample_duration: np.ndarray, estnu: STNU) -> dict[int, int]:
    """
    This function converts a sample into a dictionary that connects the samples to the keys of the contingent
    nodes in the STNU
    """
    sample = {}
    for task, duration in enumerate(sample_duration):
        find_contingent_node = estnu.translation_dict_reversed[f'{task}_{STNU.EVENT_FINISH}']
        sample[find_contingent_node] = duration
    return sample


def plot_stnu(stnu: STNU):
    G = nx.DiGraph()

    # Add nodes with their names from the translation dict
    for node in stnu.nodes:
        label = stnu.translation_dict.get(node, str(node))
        G.add_node(node, label=label)

    # Add ordinary edges
    for u, outgoing in stnu.edges.items():
        for v, edge in outgoing.items():
            if edge.weight is not None:
                G.add_edge(u, v, label=str(edge.weight), style='solid', color='black')

            # Labeled edges (UC / LC)
            if edge.uc_weight is not None:
                G.add_edge(u, v, label=f"UC {edge.uc_weight}", style='dashed', color='red')
            if edge.lc_weight is not None:
                G.add_edge(v, u, label=f"LC {-edge.lc_weight}", style='dashed', color='blue')

    # Draw graph
    pos = nx.spring_layout(G, seed=42)  # or use nx.planar_layout / nx.shell_layout

    # Draw nodes with labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # Draw edges by style
    edges_by_style = {}
    for u, v, d in G.edges(data=True):
        style = d.get('style', 'solid')
        edges_by_style.setdefault(style, []).append((u, v))

    for style, edges in edges_by_style.items():
        edge_color = [G[u][v].get('color', 'black') for (u, v) in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, style=style, edge_color=edge_color)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("STNU Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()