from pathlib import Path

from PyJobShopIntegration.problem_instances import *
import re
from typing import NamedTuple



def create_instance(file, problem_type):
    if problem_type.startswith("mmrcpsp"):
        return parse_data_rcpsp(file, problem_type)
    elif problem_type.startswith("fjsp_sdst_f"):
        return parse_data_fjsp_sdst(file, False)
    elif problem_type.startswith("fjsp_sdst"):
        return parse_data_fjsp_sdst(file)

    elif problem_type.startswith("fjsp"):
        return parse_data_fjsp(file)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


# TODO implement parser for rcpsp instances
class Mode(NamedTuple):
    job: int
    duration: int
    demands: list[int]
def parse_data_rcpsp(file, problem_type):
    # TODO define this outside the function to use for other parts of the code

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

        sink_predecessors = predecessors.pop()
        sink_successors = successors.pop()
        sink_mode = modes.pop()
        # Add predecessors for deadline tasks
        for i, (idx, deadline) in enumerate(deadlines.items()):
            mode = Mode(i + int(job_idx) - 1, deadline, [0] * (len(capacities)))
            modes.append(mode)
            # Add supersource as direct predecessor of each deadline task
            predecessors.append([0])
            successors.append([])
            successors[0].append(i + int(job_idx) - 1)
        predecessors.append(sink_predecessors)
        successors.append(sink_successors)

        modes.append(Mode(int(job_idx)+len(deadlines) - 1, 0, [0] * len(capacities)))
        # Adjust predecessors and successors for the sink task
        for i in range(len(successors)):
            if i in sink_predecessors:
                idx = successors[i].index(int(job_idx) - 1)
                successors[i][idx] = len(successors) - 1
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

def parse_data_fjsp(file_path: str) -> FJSP:
    path = Path(file_path)
    with path.open('r') as f:
        # ---- HEADER ----
        header_tokens = re.findall(r"\S+", f.readline())
        total_jobs, total_machines, _ = header_tokens
        num_jobs = int(total_jobs)
        num_machines = int(total_machines)

        # ---- RAW DATA PARSING ----
        raw_data = []
        for _ in range(num_jobs):
            tokens = re.findall(r"\S+", f.readline())
            i = 1  # skip the first token (number of ops) per original logic
            job_ops = []
            while i < len(tokens):
                mode_count = int(tokens[i])
                i += 1
                options = []
                for _ in range(mode_count):
                    machine_id = int(tokens[i]) - 1  # to 0-based
                    duration = int(tokens[i + 1])
                    options.append((machine_id, duration))
                    i += 2
                job_ops.append(options)
            raw_data.append(job_ops)

    # ---- FLATTEN OPERATIONS & BUILD INDEX MAPPING ----
    idx_map = {}
    current_idx = 0
    for j, ops in enumerate(raw_data):
        idx_map[j] = []
        for op_idx in range(len(ops)):
            idx_map[j].append(current_idx)
            current_idx += 1
    num_tasks = current_idx

    # ---- BUILD SUCCESSORS & PREDECESSORS LISTS ----
    successors = [[] for _ in range(num_tasks)]
    predecessors = [[] for _ in range(num_tasks)]
    for j, ops in enumerate(raw_data):
        for op_idx in range(len(ops) - 1):
            t_curr = idx_map[j][op_idx]
            t_next = idx_map[j][op_idx + 1]
            successors[t_curr].append(t_next)
            predecessors[t_next].append(t_curr)

    # ---- INSTANTIATE FJSP ----
    return FJSP(
        num_resources=num_machines,
        num_tasks=num_tasks,
        successors=successors,
        predecessors=predecessors,
        data=raw_data
    )


def parse_data_fjsp_sdst(file_path: str, flag=True) -> FJSP:
    # --- Read header + job_data ---
    path = Path(file_path)
    with path.open('r') as f:
        header = re.findall(r"\S+", f.readline())
        num_jobs = int(header[0])
        num_machines = int(header[1])

        job_data = []
        for _ in range(num_jobs):
            tokens = re.findall(r"\S+", f.readline())
            i = 1  # skip the “#ops” token
            ops = []
            while i < len(tokens):
                mode_count = int(tokens[i]);
                i += 1
                opts = []
                for _ in range(mode_count):
                    m_id = int(tokens[i]) - 1
                    dur = int(tokens[i + 1])
                    opts.append((m_id, dur))
                    i += 2
                ops.append(opts)
            job_data.append(ops)

        # --- Build SDST cube ---
        total_ops = sum(len(ops) for ops in job_data)
        # skip until the first numeric line
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                raise EOFError("No SDST data found in file.")
            if re.match(r'^\s*\d', line):
                f.seek(pos)
                break

        sdst = [
            [[-1] * total_ops for _ in range(total_ops)]
            for _ in range(num_machines)
        ]
        for m in range(num_machines):
            for op_idx in range(total_ops):
                line = f.readline()
                if not line:
                    raise EOFError(f"Unexpected EOF reading SDST for machine {m}, row {op_idx}")
                row = re.findall(r"-?\d+", line)
                if len(row) != total_ops:
                    raise ValueError(
                        f"Machine {m}, row {op_idx}: expected {total_ops} entries, got {len(row)}"
                    )
                sdst[m][op_idx] = [int(x) for x in row]

    # --- Flatten operations & build precedence graph ---
    idx_map = {}
    current = 0
    for j, ops in enumerate(job_data):
        idx_map[j] = []
        for _ in ops:
            idx_map[j].append(current)
            current += 1
    num_tasks = current

    successors = [[] for _ in range(num_tasks)]
    predecessors = [[] for _ in range(num_tasks)]
    for j, ops in enumerate(job_data):
        for k in range(len(ops) - 1):
            t = idx_map[j][k]
            u = idx_map[j][k + 1]
            successors[t].append(u)
            predecessors[u].append(t)

    # --- Instantiate and return FJSP-SDST ---
    return FJSPSDST(
        num_resources=num_machines,
        num_tasks=num_tasks,
        successors=successors,
        predecessors=predecessors,
        data=job_data,
        sdst_matrix = sdst if flag else []
    )