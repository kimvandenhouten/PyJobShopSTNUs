import copy
from typing import Dict
from pathlib import Path
import numpy as np
from pyjobshop import Solution
from temporal_networks.stnu import STNU
from temporal_networks.rte_star import RTEdata
from pathlib import Path

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
    start_times = []
    finish_times = []

    for task in range(num_tasks):
        start_key = f"{task}_{STNU.EVENT_START}"
        finish_key = f"{task}_{STNU.EVENT_FINISH}"

        if start_key not in estnu.translation_dict_reversed or finish_key not in estnu.translation_dict_reversed:
            # Skip tasks not included in the STNU
            continue

        node_idx_start = estnu.translation_dict_reversed[start_key]
        node_idx_finish = estnu.translation_dict_reversed[finish_key]

        start_times.append(rte_data.f[node_idx_start])
        finish_times.append(rte_data.f[node_idx_finish])

    return start_times, finish_times


def overwrite_pyjobshop_solution(solution, start_times, finish_times):
    """
    Updates a PyJobShop solution with new start/finish times.
    Assumes start_times and finish_times only apply to tasks present in the STNU.
    """
    from copy import deepcopy
    simulated_solution = deepcopy(solution)

    st_idx = 0  # index in start_times and finish_times

    for i, task in enumerate(simulated_solution.tasks):
        if st_idx >= len(start_times):
            continue
        simulated_solution.tasks[i].start = start_times[st_idx]
        simulated_solution.tasks[i].end = finish_times[st_idx]
        st_idx += 1

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
    Converts a duration sample into a mapping from each contingent‐link node
    index in estnu._contingent_nodes to its drawn duration.
    """
    sample = {}
    for task_idx, dur in enumerate(sample_duration):
        start_key = f"{task_idx}_{STNU.EVENT_START}"
        finish_key = f"{task_idx}_{STNU.EVENT_FINISH}"
        # skip any task whose nodes weren’t even in this STNU
        if start_key not in estnu.translation_dict_reversed or finish_key not in estnu.translation_dict_reversed:
            continue

        s = estnu.translation_dict_reversed[start_key]
        f = estnu.translation_dict_reversed[finish_key]
        # only sample for truly contingent links
        if (s, f) in estnu.contingent_links:
            sample[f] = int(dur)

    return sample


def check_feasibility_fjsp(
    start_times: list[int],
    durations: list[int],
    precedence_relations: list[tuple[int,int]],
    machine_chains: dict[int, list[int]]
) -> bool:
    """
    Returns True iff
      • for every (pred, succ) in precedence_relations, finish[pred] ≤ start[succ], and
      • on each machine chain, finish of each task ≤ start of the next.
    """
    finish = [start_times[i] + durations[i] for i in range(len(start_times))]

    # 1) precedence constraints
    for pred, succ in precedence_relations:
        if finish[pred] > start_times[succ]:
            return False

    # 2) resource chains (per‐machine sequences)
    for chain in machine_chains.values():
        for i in range(len(chain) - 1):
            t1, t2 = chain[i], chain[i+1]
            if finish[t1] > start_times[t2]:
                return False

    return True
