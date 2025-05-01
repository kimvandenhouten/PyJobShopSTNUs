import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from docplex.cp.model import *

import general.logger
from typing import NamedTuple
from PyJobShopIntegration.utils import get_project_root
from collections import defaultdict
logger = general.logger.get_logger(__name__)


class Mode(NamedTuple):
    job: int
    duration: int
    demands: list[int]

class MMRCPSP_CP_Benchmark:
    def __init__(self, num_tasks, num_resources, successors, predecessors, modes, capacity,
                 renewable, deadlines, noise_factor, instance_folder):

        self.num_tasks: int = num_tasks
        self.num_resources: int = num_resources
        self.successors: list[list[int]] = successors
        self.predecessors: list[list[int]] = predecessors
        self.modes: list[Mode] = modes
        self.capacity: list[int] = capacity
        self.renewable: list[bool] = renewable
        self.deadlines: dict[int, int] = deadlines
        # make durations dictionary that maps each task to its modes' durations
        self.durations: dict[int, list[int]] = {
            mode.job: [mode.duration for mode in modes if mode.job == mode.job]
            for mode in modes
        }
        # make demands dictionary that maps each task to its modes' demands
        self.needs: dict[int, list[int]] = {
            mode.job: [mode.demands for mode in modes if mode.job == mode.job]
            for mode in modes
        }
        self.noise_factor: int = noise_factor
        self.instance_folder: str = instance_folder
        self.instance_id: int = 0
        self.temporal_constraints = []

    @classmethod
    def parsche_file(cls, path, noise_factor, instance_folder):
        with open(path) as fh:
            lines = fh.readlines()

        prec_idx = lines.index("PRECEDENCE RELATIONS:\n")
        req_idx = lines.index("REQUESTS/DURATIONS:\n")
        avail_idx = lines.index("RESOURCEAVAILABILITIES:\n")
        deadlines_idx = lines.index("DEADLINES:\n")

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

        deadlines = {
            int(line.split()[0]) - 1: int(line.split()[1])
            for line in lines[deadlines_idx + 2: -1]
        }

        return cls(
            int(job_idx),
            len(capacities),
            successors,
            predecessors,
            modes,
            capacities,
            renewable,
            deadlines,
            noise_factor,
            instance_folder
        )


    def solve_with_warmstart(self, durations=None, time_limit=None, write=False, output_file="results.csv", mode="Quiet",
              initial_solution=None):

        # Set durations to self.durations if no input vector is given
        durations = self.durations if durations is None else durations
        demands = self.needs
        capacities = self.capacity
        nb_tasks = len(self.durations)
        nb_resources = len(capacities)

        # Create model
        mdl = CpoModel()

        # Create task interval variables
        tasks = [interval_var(name='T{}'.format(i + 1), size=durations[i]) for i in range(nb_tasks)]

        # Add precedence constraints
        if self.problem_type == "RCPSP":
            mdl.add(start_of(tasks[s]) >= end_of(tasks[t]) for t in range(nb_tasks) for s in self.successors[t])

        elif self.problem_type == "RCPSP_max":
            mdl.add(start_of(tasks[s]) + lag <= start_of(tasks[t]) for (s, lag, t) in self.temporal_constraints)


        else:
            raise NotImplementedError(f"Problem type has not been recognized {self.problem_type}")

        # Constrain capacity of needs
        mdl.add(sum(pulse(tasks[t], demands[t][r]) for t in range(nb_tasks) if demands[t][r] > 0) <= capacities[r]
                for r in range(nb_resources))

        # Add objective value
        mdl.add(minimize(max(end_of(t) for t in tasks)))

        # Apply initial solution if provided
        if initial_solution:
            starting_point = CpoModelSolution()
            for i in range(nb_tasks):
                task_name = f'T{i}'
                if task_name in initial_solution:
                    task_start = initial_solution[task_name]['start']
                    task_end = initial_solution[task_name]['end']
                    starting_point.add_interval_var_solution(tasks[i], start=task_start, end=task_end)
            mdl.set_starting_point(starting_point)

        # Solve model
        logger.info('Solving model...')

        if mode == "Quiet":
            res = mdl.solve(TimeLimit=time_limit, Workers=1, LogVerbosity="Quiet")
        else:
            res = mdl.solve(TimeLimit=time_limit, Workers=1)

        data = []
        if res:
            for i in range(len(self.durations)):
                start = res.get_var_solution(tasks[i]).start
                end = res.get_var_solution(tasks[i]).end
                data.append({"task": i, "start": start, "end": end, })
            data_df = pd.DataFrame(data)
            if write:
                data_df.to_csv(output_file)
        else:
            logger.info('WARNING: CP solver failed')
            data_df = None

        return res, data_df

    def solve(self, durations, time_limit=None, write=False, output_file="results.csv", mode="Quiet"):
        capacities = self.capacity
        nb_tasks = self.num_tasks
        nb_resources = self.num_resources
        all_modes = self.modes
        modes_by_task = defaultdict(list)
        for m in all_modes:
            modes_by_task[m.job].append(m)
        modes_by_task = dict(modes_by_task)
        successors = self.successors  # Dict[int, List[int]]
        self.durations = durations if self.durations is None else self.durations

        mdl = CpoModel()

        task_mode_intervals = []
        # Step 1: Create optional interval variables per mode
        for i in range(nb_tasks):
            # Create interval variables for each mode of the task
            mode_intervals = []
            for j, mode in enumerate(modes_by_task[i]):
                dur = mode.duration
                res_demand = mode.demands
                ivar = interval_var(size=dur, name=f"T{i}_M{j}", optional=True)
                mode_intervals.append((ivar, res_demand))
            task_mode_intervals.append(mode_intervals)
            # Step 2: Exactly one mode is selected → exactly one interval is present
            mdl.add(sum(presence_of(ivar) for (ivar, _) in mode_intervals) == 1)

        # # Step 3: Add precedence constraints (over all possible pairs of selected intervals)
        # for t in range(nb_tasks):
        #     for s in successors[t]:
        #         for (iv_t, _) in task_mode_intervals[t]:
        #             for (iv_s, _) in task_mode_intervals[s]:
        #                 mdl.add(start_of(iv_s) >= end_of(iv_t))
        # Step 4: Add resource usage constrain
        mdl.add(
            sum(pulse(iv, demand[r]) for i in range(nb_tasks) for (iv, demand) in task_mode_intervals[i] if
                demand[r] > 0) <= capacities[r]
            for r in range(nb_resources)
        )

        # Step 5: Minimize makespan
        all_ivars = [iv for modes in task_mode_intervals for (iv, _) in modes]
        mdl.add(minimize(max(end_of(iv) for iv in all_ivars)))

        # Step 6: Solve model
        logger.info('Solving multi-mode RCPSP model...')

        if mode == "Quiet":
            res = mdl.solve(TimeLimit=time_limit, Workers=1, LogVerbosity="Quiet")
        else:
            res = mdl.solve(TimeLimit=time_limit, Workers=1)

        data = []
        if res:
            for i in range(nb_tasks):
                for j, (iv, _) in enumerate(task_mode_intervals[i]):
                    sol = res.get_var_solution(iv)
                    if sol.is_present():
                        data.append({
                            "task": i,
                            "mode": j,
                            "start": sol.start,
                            "end": sol.end,
                        })
            data_df = pd.DataFrame(data)
            if write:
                data_df.to_csv(output_file)
        else:
            logger.warning("WARNING: CP solver failed.")
            data_df = None

        return res, data_df

    def solve_reactive(self, durations, scheduled_start_times, current_time, time_limit=None, initial_solution=None):

        # Set durations to self.durations if no input vector is given
        durations = self.durations if durations is None else durations
        demands = self.needs
        capacities = self.capacity
        nb_tasks = len(self.durations)
        nb_resources = len(capacities)

        # Create model
        mdl = CpoModel()

        # Create task interval variables
        tasks = [interval_var(name='T{}'.format(i + 1), size=durations[i]) for i in range(nb_tasks)]

        # Add precedence constraints
        if self.problem_type == "RCPSP":
            mdl.add(start_of(tasks[s]) >= end_of(tasks[t]) for t in range(nb_tasks) for s in self.successors[t])

        elif self.problem_type == "RCPSP_max":
            mdl.add(start_of(tasks[s]) + lag <= start_of(tasks[t]) for (s, lag, t) in self.temporal_constraints)

        else:
            raise NotImplementedError(f"Problem type has not been recognized {self.problem_type}")

        # Constraint to enforce already scheduled start times (rescheduling approach)
        for t in range(nb_tasks):
            if scheduled_start_times[t] >= 0:
                mdl.add(start_of(tasks[t]) == scheduled_start_times[t])
            else:
                mdl.add(start_of(tasks[t]) >= current_time)

        # Constrain capacity of needs
        mdl.add(sum(pulse(tasks[t], demands[t][r]) for t in range(nb_tasks) if demands[t][r] > 0) <=
                capacities[r] for r in range(nb_resources))

        # Add objective value
        mdl.add(minimize(max(end_of(t) for t in tasks)))

        # Apply initial solution if provided
        if initial_solution:
            logger.info(f'Set warm start to {initial_solution}')
            starting_point = CpoModelSolution()
            for i in range(nb_tasks):
                task_name = f'T{i}'
                if task_name in initial_solution:
                    task_start = initial_solution[i]
                    task_end = task_start + durations[i]
                    starting_point.add_interval_var_solution(tasks[i], start=task_start, end=task_end)
            mdl.set_starting_point(starting_point)

        # Solve model
        logger.info('Solving model...')
        res = mdl.solve(TimeLimit=time_limit, Workers=1, LogVerbosity="Quiet")

        start_times = []
        if res:
            for i in range(len(self.durations)):
                start = res.get_var_solution(tasks[i]).start
                start_times.append(start)
            makespan = res.solution.get_objective_value()
            logger.info(f'Makespan is {makespan}')
            logger.info(f'Solve time reactive is {res.get_solve_time()}')
            return start_times, makespan
        else:
            logger.info('WARNING: CP solver failed')
            return None, np.inf

    def solve_saa(self, durations, time_limit=None,  write=False, output_file="results.csv"):
        # Create model
        mdl = CpoModel()
        demands = self.needs
        capacities = self.capacity
        nb_tasks = len(self.durations)
        nb_resources = len(capacities)

        nb_scenarios = len(durations)
        scenarios = range(nb_scenarios)

        # Create task interval variables
        all_tasks = []
        first_stage = [mdl.integer_var(name=f'start_times_{i}') for i in range(nb_tasks)]
        makespans = [mdl.integer_var(name=f'makespan_scenarios{omega}') for omega in scenarios]

        mdl.add(first_stage[t] >= 0 for t in range(nb_tasks))
        # Make scenario intervals
        for omega in range(nb_scenarios):
            tasks = [mdl.interval_var(name=f'T{i}_{omega}', size=durations[omega][i]) for i in range(nb_tasks)]
            all_tasks.append(tasks)

        # Add constraints
        for omega in scenarios:
            tasks = all_tasks[omega]

            # Add relation between scenario start times and first stage decision
            mdl.add(start_of(tasks[t]) == first_stage[t] for t in range(nb_tasks))

            # Precedence relations
            if self.problem_type == "RCPSP":
                mdl.add(start_of(tasks[s]) >= end_of(tasks[t]) for t in range(nb_tasks) for s in self.successors[t])

            elif self.problem_type == "RCPSP_max":
                mdl.add(start_of(tasks[s]) + lag <= start_of(tasks[t]) for (s, lag, t) in self.temporal_constraints)

            else:
                raise NotImplementedError(f"Problem type has not been recognized {self.problem_type}")

            # Constrain capacity of resources
            mdl.add(
                sum(pulse(tasks[t], demands[t][r]) for t in range(nb_tasks) if demands[t][r] > 0) <= capacities[r] for r
                in
                range(nb_resources))

            # Makespan constraint for this scenario
            mdl.add(makespans[omega] >= max(end_of(t) for t in tasks))

        # Solve model, objective is Sample Average Approximation of the makespan
        mdl.add(minimize(sum([makespans[omega] for omega in scenarios])))

        res = mdl.solve(TimeLimit=time_limit, Workers=1, LogVerbosity="Quiet")

        if res:
            start_times = [res.get_var_solution(first_stage[i]).value for i in range(nb_tasks)]
        else:
            logger.warning('WARNING: CP solver failed')
            start_times = None

        return res, start_times

    def get_bound(self, mode="upper_bound"):
        lb = []
        ub = []
        for duration in self.durations:
            if duration == 0:
                lb.append(0)
                ub.append(0)
            else:
                lower_bound = int(max(1, duration - self.noise_factor * np.sqrt(duration)))
                upper_bound = int(duration + self.noise_factor * np.sqrt(duration))
                if lower_bound == upper_bound:
                    upper_bound += 1
                lb.append(lower_bound)
                ub.append(upper_bound)
        if mode == "upper_bound":
            return ub
        else:
            return lb

    def sample_durations(self, nb_scenarios=1):
        scenarios = []
        lower_bound = self.get_bound("lower_bound")
        upper_bound = self.get_bound("upper_bound")
        for _ in range(nb_scenarios):
            scenario = []
            for i in range(0, len(self.durations)):
                if lower_bound[i] == 0 and upper_bound[i] == 0:
                    duration_sample = 0
                else:
                    duration_sample = np.random.randint(lower_bound[i], upper_bound[i])
                scenario.append(duration_sample)
            scenarios.append(scenario)

        return scenarios

    def update_needs(self, schedule):
        needs = []
        for i, task in enumerate(schedule):
            mode = task['mode']
            needs.append(self.modes[mode].demands)
        self.needs = needs

