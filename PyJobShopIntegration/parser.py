from pathlib import Path

from PyJobShopIntegration.problem_instances import *
import re
from typing import NamedTuple



def create_instance(file, problem_type, sdst=False):
    if problem_type.startswith("mmrcpsp"):
        return parse_data_rcpsp(file, problem_type)

    elif problem_type.startswith("fjspgtl"):
        # return parse_data_fjsp_nw_and_gtl(file, problem_type)
        num_machines, data = parse_data_fjsp(file)
        return build_model_fjsp_gtl(num_machines, data)


    elif problem_type.startswith("fjspnw"):
        # return parse_data_fjsp_nw_and_gtl(file, problem_type)
        num_machines, data = parse_data_fjsp(file)
        return build_model_fjsp_nw(num_machines, data)

    elif problem_type.startswith("fjsp"):
        if sdst:
            num_machines, data, sdst_matrix = parse_data_fjsp_sdst(file)
        else:
            num_machines, data = parse_data_fjsp(file)
            sdst_matrix = None
        return build_model_fjsp(num_machines, data, sdst_matrix)
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

def parse_data_fjsp_nw_and_gtl(file, problem_type):
    path = Path(file)
    with open(path, 'r') as f:
        # Read header line
        header_tokens = re.findall(r"\S+", f.readline())
        total_jobs, total_machines, _ = header_tokens
        num_jobs = int(total_jobs)
        num_machines = int(total_machines)
        num_tasks = 0

        data = []
        # Parse each job line
        for _ in range(num_jobs):
            line = f.readline()
            parsed = re.findall(r"\S+", line)
            i = 1  # skip first token per original logic
            job_ops = []

            while i < len(parsed):
                num_tasks += 1
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

    if problem_type.endswith("nw"):
        return FJSPNW(
            num_tasks,
            num_machines,
            data
        )

    elif problem_type.endswith("gtl"):
        return FJSPGTL(
            num_tasks,
            num_machines,
            data
        )

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")



def parse_data_fjsp_sdst(file_path):
        path = Path(file_path)
        with open(path, 'r') as f:
            # 1) Header: total_jobs total_machines max_ops_per_job
            header = re.findall(r"\S+", f.readline())
            num_jobs = int(header[0])
            num_machines = int(header[1])

            # 2) Per-job operation options
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

            # 4) Build empty SDST cube
            total_ops = sum(len(ops) for ops in job_data)
            sdst = [
                [[-1] * total_ops for _ in range(total_ops)]
                for _ in range(num_machines)
            ]

            # 5) Skip any blank or header lines until the first numeric row
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    raise EOFError("No SDST data found in file.")
                if re.match(r'^\s*\d', line):
                    # rewind one line so the numeric row will be read below
                    f.seek(pos)
                    break

            # 6) Now read exactly `num_machines` blocks of `total_ops` rows each
            for m in range(num_machines):
                for i in range(total_ops):
                    line = f.readline()
                    if not line:
                        raise EOFError(f"Unexpected EOF reading SDST for machine {m}, row {i}")
                    row = re.findall(r"-?\d+", line)
                    if len(row) != total_ops:
                        raise ValueError(
                            f"Machine {m}, row {i}: expected {total_ops} entries, got {len(row)}"
                        )
                    sdst[m][i] = [int(x) for x in row]

            return num_machines, job_data, sdst

def build_model_fjsp(num_machines, data, sdst_matrix=None):
    """
    Build a PyJobShop Model:
      - Adds machines, jobs, tasks, modes
      - Chains each job’s ops with end-before-start
      - If sdst_matrix is provided, registers sequence-dependent setup times
    """
    model = Model()

    # A) machines
    machines = [
        model.add_machine(name=f"Machine {m}")
        for m in range(num_machines)
    ]

    # B) create jobs & tasks
    tasks = {}
    for j_idx, ops in enumerate(data):
        job = model.add_job(name=f"Job {j_idx}")
        for o_idx in range(len(ops)):
            tasks[(j_idx, o_idx)] = model.add_task(job, name=f"Task ({j_idx},{o_idx})")

    # C) add modes and chain precedence
    for j_idx, ops in enumerate(data):
        for o_idx, options in enumerate(ops):
            task = tasks[(j_idx, o_idx)]
            for m_id, dur in options:
                model.add_mode(task, machines[m_id], duration=dur)
        for o_idx in range(len(ops)-1):
            model.add_end_before_start(
                tasks[(j_idx, o_idx)],
                tasks[(j_idx, o_idx+1)]
            )

    # D) optional: register sequence‐dependent setup times
    if sdst_matrix is not None:
        # flatten tasks in the same file‐order the parser used
        flat_tasks = [
            tasks[(j_idx, o_idx)]
            for j_idx, ops in enumerate(data)
            for o_idx in range(len(ops))
        ]

        for m_idx, machine in enumerate(machines):
            for i, ti in enumerate(flat_tasks):
                for j, tj in enumerate(flat_tasks):
                    setup_time = sdst_matrix[m_idx][i][j]
                    model.add_setup_time(machine, ti, tj, duration=setup_time)

    return model

def build_model_fjsp_nw(num_machines, data):
    """
    Build a PyJobShop Model:
      - Adds machines, jobs, tasks, modes
      - Chains each job’s ops with end-before-start
    """
    model = Model()

    # A) machines
    machines = [
        model.add_machine(name=f"Machine {m}")
        for m in range(num_machines)
    ]

    # B) create jobs & tasks
    tasks = {}
    for j_idx, ops in enumerate(data):
        job = model.add_job(name=f"Job {j_idx}")
        for o_idx in range(len(ops)):
            tasks[(j_idx, o_idx)] = model.add_task(job, name=f"Task ({j_idx},{o_idx})")

    # C) add modes and chain precedence and no-wait constraint
    for j_idx, ops in enumerate(data):
        for o_idx, options in enumerate(ops):
            task = tasks[(j_idx, o_idx)]
            for m_id, dur in options:
                model.add_mode(task, machines[m_id], duration=dur)
        for o_idx in range(len(ops)-1):
            model.add_end_before_start(
                tasks[(j_idx, o_idx)],
                tasks[(j_idx, o_idx + 1)]
            )
            # model.add_start_before_end(
            #     tasks[(j_idx, o_idx + 1)],
            #     tasks[(j_idx, o_idx)],
            #     0
            # )

    return model

def build_model_fjsp_gtl(num_machines, data):
    """
    Build a PyJobShop Model:
      - Adds machines, jobs, tasks, modes
      - Chains each job’s ops with end-before-start
    """
    model = Model()

    # A) machines
    machines = [
        model.add_machine(name=f"Machine {m}")
        for m in range(num_machines)
    ]

    # B) create jobs & tasks
    tasks = {}
    for j_idx, ops in enumerate(data):
        job = model.add_job(name=f"Job {j_idx}")
        for o_idx in range(len(ops)):
            tasks[(j_idx, o_idx)] = model.add_task(job, name=f"Task ({j_idx},{o_idx})")

    # C) add modes and chain precedence and generalized time-lags
    for j_idx, ops in enumerate(data):
        for o_idx, options in enumerate(ops):
            task = tasks[(j_idx, o_idx)]
            for m_id, dur in options:
                model.add_mode(task, machines[m_id], duration=dur)
        for o_idx in range(len(ops)-1):
            model.add_end_before_start(
                tasks[(j_idx, o_idx)],
                tasks[(j_idx, o_idx + 1)],
                0
            )
            model.add_start_before_end(
                tasks[(j_idx, o_idx + 1)],
                tasks[(j_idx, o_idx)],
                0
            )

    return model