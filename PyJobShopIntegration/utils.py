
from typing import Dict
from pyjobshop import Solution
from temporal_networks.stnu import STNU


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
    unique_tuples = []
    seen = set()

    for current_tuple in tuples_list:
        if current_tuple not in seen:
            unique_tuples.append(current_tuple)
            seen.add(current_tuple)

    return unique_tuples


def get_resource_chains(schedule, capacity, resources, complete=False):
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


def add_resource_chains(stnu, resource_chains):
    for pred_task, succ_task in resource_chains:
        # the finish of the predecessor should precede the start of the successor
        pred_idx_finish = stnu.translation_dict_reversed[
            f"{pred_task}_{STNU.EVENT_FINISH}"]  # Get translation index from finish of predecessor
        suc_idx_start = stnu.translation_dict_reversed[
            f"{succ_task}_{STNU.EVENT_START}"]  # Get translation index from start of successor

        # add constraint between predecessor and successor
        stnu.set_ordinary_edge(suc_idx_start, pred_idx_finish, 0)

    return stnu
