def group_shift_solution_resequenced(solution, model, buffer_time=1, split_time=25):
    """
    Shift the 'right cluster' of tasks to the first safe point, and resequence them on machines.
    """
    """
        Translate the entire right cluster (tasks starting after split_time) to shift_start_time.
        """
    # Step 1: Find earliest start in right cluster
    right_starts = [
        task.start for idx, task in enumerate(solution.tasks)
        if task.start >= split_time and "Deadline" not in model.tasks[idx].name
    ]

    if not right_starts:
        return solution

    old_right_start = min(right_starts)
    print(f"[DEBUG] Old right cluster start: {old_right_start}")

    # Step 2: Find max early end
    all_task_ends = [
        task.end for idx, task in enumerate(solution.tasks)
        if task.start < split_time and "Deadline" not in model.tasks[idx].name
    ]

    if not all_task_ends:
        return solution

    max_early_end = max(all_task_ends)
    print(f"[DEBUG] Max early finish time: {max_early_end}")

    shift_start_time = max_early_end + buffer_time
    print(f"[DEBUG] Desired new start time: {shift_start_time}")

    # Step 3: Calculate shift amount
    shift_amount = old_right_start - shift_start_time
    print(f"[DEBUG] Shifting late tasks by: {shift_amount}")

    # Step 4: Shift
    for idx, task in enumerate(solution.tasks):
        if task.start >= split_time and "Deadline" not in model.tasks[idx].name:
            task.start -= shift_amount
            task.end -= shift_amount

    return solution