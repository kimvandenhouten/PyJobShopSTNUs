def check_deadline_feasibility(raw_data, job_deadlines):
    """
    Checks whether each job can theoretically meet its deadline based on minimum task durations.

    Args:
        raw_data: List of jobs, each job is a list of tasks, each task has a list of (duration, machine_id) tuples.
        job_deadlines: Dict from job index to deadline.

    Returns:
        List of tuples: (job_idx, required_min_duration, deadline)
    """
    infeasible_jobs = []
    for job_idx, job_tasks in enumerate(raw_data):
        min_total_duration = sum(min(d for d, _ in task) for task in job_tasks)
        deadline = job_deadlines[job_idx]
        if min_total_duration > deadline:
            infeasible_jobs.append((job_idx, min_total_duration, deadline))
    return infeasible_jobs


def compute_slack_weights(raw_data, job_deadlines, high=10, medium=5, low=1, slack_thresholds=(3, 10)):
    """
    Assigns urgency weights to jobs based on slack between their minimum duration and deadline.

    Args:
        raw_data: Same format as check_deadline_feasibility.
        job_deadlines: Dict of job deadlines.
        high, medium, low: Weights to assign based on slack.
        slack_thresholds: (low_slack, high_slack) boundaries.

    Returns:
        Dict of job_idx -> weight
    """
    weights = {}
    for job_idx, job_tasks in enumerate(raw_data):
        min_total_duration = sum(min(d for d, _ in task) for task in job_tasks)
        slack = max(0, job_deadlines[job_idx] - min_total_duration)

        if slack <= slack_thresholds[0]:
            weights[job_idx] = high
        elif slack <= slack_thresholds[1]:
            weights[job_idx] = medium
        else:
            weights[job_idx] = low
    return weights



def estimate_deadline_violation_probabilities(solution, model, sampler, job_deadlines, num_samples=1000):
    """
    Estimate probability of deadline violation per job using sampled durations.

    Parameters:
    - solution: Static schedule solution from PyJobShop.
    - model: The CP model.
    - sampler: An instance of DiscreteUniformSampler.
    - job_deadlines: Dict mapping job indices to deadline values.
    - num_samples: Number of duration samples to draw.

    Returns:
    - Dict[job_idx] = estimated violation probability
    """
    job_violations = {j: 0 for j in job_deadlines}

    for _ in range(num_samples):
        durations = sampler.sample()

        # Build simulated task end times
        task_end_times = [0] * len(solution.tasks)

        for idx, task in enumerate(solution.tasks):
            start = task.start
            duration = durations[idx]
            end = start + duration
            task_end_times[idx] = end

        # Check job deadline satisfaction
        for job_idx, deadline in job_deadlines.items():
            last_task_idx = max(
                idx for idx, task in enumerate(model.tasks)
                if task.job == job_idx and "Deadline" not in task.name
            )
            if task_end_times[last_task_idx] > deadline:
                job_violations[job_idx] += 1

    return {job: job_violations[job] / num_samples for job in job_violations}
