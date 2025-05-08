from pathlib import Path

from PyJobShopIntegration.problem_instances import *
import re
from typing import NamedTuple

def create_instance(file, problem_type):
    if problem_type.startswith("mmrcpsp"):
        # existing RCPSP parser; returns appropriate object
        return parse_data_rcpsp(file, problem_type)
    elif problem_type.startswith("fjsp"):
        num_machines, data = parse_data_fjsp(file)
        return build_model_fjsp(num_machines, data)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


# TODO implement parser for rcpsp instances
def parse_data_rcpsp(file, problem_type):
    class Mode(NamedTuple):
        job: int
        duration: int
        demands: list[int]
    with open(file) as fh:
        lines = fh.readlines()

    prec_idx = lines.index("PRECEDENCE RELATIONS:\n")
    req_idx = lines.index("REQUESTS/DURATIONS:\n")
    avail_idx = lines.index("RESOURCEAVAILABILITIES:\n")

    successors = []

    for line in lines[prec_idx + 2: req_idx - 1]:
        _, _, _, _, *jobs, _ = re.split(r"\s+", line)
        successors.append([int(x) - 1 for x in jobs])

    predecessors: list[list[int]] = [[] for _ in range(len(successors))]
    for job in range(len(successors)):
        for succ in successors[job]:
            predecessors[succ].append(job)

    mode_data = [
        re.split(r"\s+", line.strip())
        for line in lines[req_idx + 3: avail_idx - 1]
    ]

    # Prepend the job index to mode data lines if it is missing.
    for idx in range(len(mode_data)):
        if idx == 0:
            continue

        prev = mode_data[idx - 1]
        curr = mode_data[idx]

        if len(curr) < len(prev):
            curr = prev[:1] + curr
            mode_data[idx] = curr

    modes = []
    for mode in mode_data:
        job_idx, _, duration, *consumption = mode
        demands = list(map(int, consumption))
        modes.append(Mode(int(job_idx) - 1, int(duration), demands))

    _, *avail, _ = re.split(r"\s+", lines[avail_idx + 2])
    capacities = list(map(int, avail))

    renewable = [
        x == "R"
        for x in lines[avail_idx + 1].strip().split(" ")
        if x in ["R", "N"]  # R: renewable, N: non-renewable
    ]
    if problem_type.endswith("d"):
        deadlines_idx = lines.index("DEADLINES:\n")
        deadlines = {
            int(line.split()[0]) - 1: int(line.split()[1])
            for line in lines[deadlines_idx + 2: -1]
        }
        return MMRCPSPD(
            int(job_idx),
            len(capacities),
            successors,
            predecessors,
            modes,
            capacities,
            renewable,
            deadlines,
        )
    elif problem_type.endswith("gtl"):
        # TODO implement parsing the gtl data
        args = []
        return MMRCPSPGTL(
            int(job_idx),
            len(capacities),
            successors,
            predecessors,
            modes,
            capacities,
            renewable,
            args,
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

def parse_data_fjsp(file):
    path = Path(file)
    with open(path, 'r') as f:
        # Read header line
        header_tokens = re.findall(r"\S+", f.readline())
        total_jobs, total_machines, _ = header_tokens
        num_jobs = int(total_jobs)
        num_machines = int(total_machines)

        data = []
        # Parse each job line
        for _ in range(num_jobs):
            line = f.readline()
            parsed = re.findall(r"\S+", line)
            i = 1  # skip first token per original logic
            job_ops = []

            while i < len(parsed):
                mode_count = int(parsed[i])
                i += 1
                options = []
                for _ in range(mode_count):
                    machine_id = int(parsed[i]) - 1  # to 0-based
                    duration = int(parsed[i + 1])
                    options.append((machine_id, duration))
                    i += 2
                job_ops.append(options)

            data.append(job_ops)

    return num_machines, data

def build_model_fjsp(num_machines, data):
    # Construct the model
    model = Model()

    # Add machines
    machines = [
        model.add_machine(name=f"Machine {idx}")
        for idx in range(num_machines)
    ]

    # Prepare job and task containers
    jobs = {}
    tasks = {}

    # First pass: create jobs and task placeholders
    for job_idx, job_data in enumerate(data):
        job = model.add_job(name=f"Job {job_idx}")
        jobs[job_idx] = job
        for idx in range(len(job_data)):
            task_key = (job_idx, idx)
            tasks[task_key] = model.add_task(job, name=f"Task {task_key}")

    # Second pass: add modes and precedence
    for job_idx, job_data in enumerate(data):
        # Add mode options for each task
        for idx, task_data in enumerate(job_data):
            task_key = (job_idx, idx)
            task = tasks[task_key]
            for machine_id, duration in task_data:
                machine = machines[machine_id]
                model.add_mode(task, machine, duration)

        # Add precedence constraints within the job. Not sure to keep this, also in example.
        for idx in range(len(job_data) - 1):
            first = tasks[(job_idx, idx)]
            second = tasks[(job_idx, idx + 1)]
            model.add_end_before_start(first, second)

    return model