import numpy as np
import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import general.logger
logger = general.logger.get_logger(__name__)


class RCPSP_MIP_Benchmark:
    def __init__(self, capacity, durations, successors, needs, temporal_constraints=None, problem_type="RCPSP",
                 instance_folder="", instance_id="", noise_factor=1):
        # convert to RCPSP instance
        self.capacity = capacity
        self.durations = durations
        self.needs = needs
        self.successors = successors
        self.temporal_constraints = temporal_constraints
        self.problem_type = problem_type
        self.instance_folder = instance_folder
        self.instance_id = instance_id
        self.num_tasks = len(self.durations)
        self.noise_factor = noise_factor

    @classmethod
    def parsche_file(cls, directory, instance_folder, instance_id, noise_factor):

        if instance_folder[0] == "j":
            filename = f'{directory}/{instance_folder}/PSP{instance_id}.SCH'
        elif instance_folder[0:3] == "ubo":
            filename = f'{directory}/{instance_folder}/psp{instance_id}.sch'
        else:
            raise ValueError(f"instance folder is not recognized ({instance_folder})")

        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract the header information
        header = lines[0].strip().split()
        n_tasks = int(header[0])
        n_res = int(header[1])

        # Initialize structures
        durations = [0] * (n_tasks + 2)  # Assuming tasks are numbered from 0 to n_tasks + 1
        needs = []
        temporal_relations = []

        # Parse each task line
        for line in lines[1:n_tasks + 2]:
            parts = line.strip().split()
            task_id = int(parts[0])
            num_successors = int(parts[2])
            successors = parts[3: 3 + num_successors]
            lags = parts[3 + num_successors:]
            for i, suc in enumerate(successors):
                eval_lags = lags[i]
                eval_lags = eval_lags.strip('[]').split(',')
                eval_lags = [int(i) for i in eval_lags]
                for lag in eval_lags:
                    temporal_relations.append((task_id, int(lag), int(suc)))

        for line in lines[n_tasks + 3:-1]:
            parts = line.strip().split()
            task_id = int(parts[0])
            duration = int(parts[2])
            durations[task_id] = duration
            resource_needs = parts[3:]
            resource_needs = [int(i) for i in resource_needs]
            needs.append(resource_needs)

        # Resource capacities and the last resource line
        capacity = list(map(int, lines[-1].strip().split()))

        rcpsp_max = cls(capacity, durations, None, needs, temporal_relations, "RCPSP_max",
                        instance_folder, instance_id, noise_factor)

        return rcpsp_max

    def solve_MIP(self, durations=None, time_limit=None):

        # Set durations to self.durations if no input vector is given
        durations = self.durations if durations is None else durations
        demands = self.needs
        capacities = self.capacity
        nb_tasks = len(self.durations)
        nb_resources = len(capacities)

        # Create a set for the tasks
        self.J = set([j for j in range(0, nb_tasks)])

        # Create a set for the resoruces
        self.R = set([r for r in range(0, nb_resources)])

        # Create a parameter for the durations
        self.p = {j: durations[j] for j in self.J}

        # Create a parameter for the demands levels
        self.l = {(r, j): demands[j][r] for r in self.R for j in self.J}

        # Create a parameter for the capacity maxima
        self.b = {r: self.capacity[r] for r in self.R}

        # The MIP-model is a time-indexed model
        self.T = np.array(range(0, int(sum(self.durations) * 1.5)))

        # MODEL: create concrete model
        self.model = ConcreteModel()

        # MODEL: variables
        self.model.t = Var(self.J, domain=NonNegativeIntegers)  # finish time of j
        self.model.x = Var(self.J, self.T, domain=Boolean)  # j starts at time t
        self.model.u = Var(self.R, self.T, domain=Boolean)  # j starts at time t
        self.model.fM = Var(domain=NonNegativeIntegers)

        # MODEL: objective
        self.model.Obj = Objective(expr=self.model.fM, sense=minimize)

        self.model.cons = ConstraintList()

        # MODEL: makespan constraints
        for j in self.J:
            self.model.cons.add(self.model.t[j] <= self.model.fM)

        # MODEL: finish time constraints
        for j in self.J:
            self.model.cons.add(sum([self.model.x[j, t] * t for t in self.T]) + self.p[j] == self.model.t[j])

        # MODEL: temporal constraint
        # TODO: correctly implement the temporal constraints
        for (pred, lag, suc) in self.temporal_constraints:
            self.model.cons.add(self.model.t[pred] - self.p[pred] + lag <= self.model.t[suc] - self.p[suc])

        # MODEL: resource constraint
        for t in self.T:
            for r in self.R:
                lhs = 0
                if sum([self.l[r, j] for j in self.J]) > 0:
                    for j in self.J:
                        for tau in self.T[self.T <= t]:
                            if tau >= t - self.p[j] + 1:
                                lhs += self.l[r, j] * self.model.x[j, tau]
                    self.model.cons.add(lhs <= self.b[r])

        # MODEL: no negative start times constraint
        for j in self.J:
            self.model.cons.add(self.model.t[j] - self.p[j] >= 0)

        # MODEL: only 1 start time constraint
        for j in self.J:
            self.model.cons.add(sum([self.model.x[j, t] for t in self.T]) == 1)

        # TODO: implement the CPLEX version
        solver = 'cplex'
        opt = SolverFactory(solver)
        if time_limit is not None:
            opt.options['timelimit'] = time_limit
            print(f'Set time limit to {time_limit}')
        solution = opt.solve(self.model)

        self.terminal = solution.solver.termination_condition
        self.solver_time = round(solution.solver.time, 4)
        logger.info(f'Solve time is {self.solver_time} seconds')
        logger.info(f'Solver status is {self.terminal}')
        if self.terminal == "optimal" or self.terminal == "feasible":
            self.objective = round(self.model.fM(), 0)
            logger.info(f'Objective is {self.objective}')
        else:
            self.objective = None

        return self.solver_time, self.terminal, self.objective

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



