from problem_instances import *
import re
from typing import NamedTuple
def parse_data(file, problem_type):
    if problem_type.startswith("mmrcpsp"):
        return parse_data_rcpsp(file, problem_type)
    elif problem_type.startswith("fjsp", problem_type):
        return parse_data_fjsp(file)

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
    if problem_type.endswith("wd"):
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
# TODO implement parser for fjsp instances
def parse_data_fjsp(file):
    pass