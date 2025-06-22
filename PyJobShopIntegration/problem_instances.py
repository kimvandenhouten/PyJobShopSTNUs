from typing import NamedTuple

import numpy as np
from docplex.cp.model import *
from pyjobshop import Model, MAX_VALUE, Task

from PyJobShopIntegration.Sampler import DiscreteUniformSampler

np.random.seed(42)  # For reproducibility
# Parent class of all instances, could include more important methods if needed
class Instance():

    def __init__(self, num_tasks, num_resources, successors, predecessors):
        """
        Initialize the instance.

        :param num_tasks: Number of tasks in the project.
        :param num_resources: Number of resources available.
        :param successors: List of successors for each task.
        :param predecessors: List of predecessors for each task.
        """
        self.num_tasks = num_tasks
        self.num_resources = num_resources
        self.successors = successors
        self.predecessors = predecessors
        self.model = None
    def get_objective_rte(self, rte_data, objective="makespan"):
        """
        Get the objective value from the RTE data.

        :param rte_data: The RTE data containing the results.
        :param objective: The type of objective to retrieve (default is "makespan").
        :return: The objective value.
        """
        if objective == "makespan":
            return max(rte_data.f.values())
        else:
            raise ValueError("Unknown objective type.")

    def get_objective(self, result_tasks, objective="makespan"):
        """
        Get the objective value from the result tasks.

        :param result_tasks: The result tasks containing the results.
        :param objective: The type of objective to retrieve (default is "makespan").
        :return: The objective value.
        """
        if objective == "makespan":
            return max(task.end for task in result_tasks)
        else:
            raise ValueError("Unknown objective type.")

    def check_duration_feasibility(self, start_times, finish_times, durations):
        """
        Check the duration feasibility of the tasks.
        :param start_times: Start times of the tasks.
        :param finish_times: Finish times of the tasks.
        :param durations: Durations of the tasks.
        :return: True if feasible, False otherwise.
        """
        for (job, dur) in enumerate(durations):
            if finish_times[job] - start_times[job] != dur or dur < 0:
                return False
        return True

    def check_precedence_feasibility(self, start_times, finish_times, successors):
        """
        Check the precedence feasibility of the tasks.
        :param start_times: Start times of the tasks.
        :param finish_times: Finish times of the tasks.
        :param successors: Successors of the tasks.
        :return: True if feasible, False otherwise.
        """
        for (job, job_successors) in enumerate(successors):
            for suc in job_successors:
                if finish_times[suc] < start_times[job]:
                    return True
        return True

    def check_feasibility(self, start_times, finish_times, *args):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_sample_length(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_bounds(self, noise_factor):
        """
        Get the bounds for the durations.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_schedule(self, result_tasks):
        """
        Get the schedule for the tasks.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def solve_reactive(self, *args):
        """
        Solve the problem using a reactive approach.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    # TODO this might not work for single mode
    def get_real_durations(self, result_tasks, duration_sample):
        """
        Get the real durations for the tasks.
        This method should be implemented in subclasses.
        """
        real_durations = []
        for task in result_tasks:
            mode = task.mode
            real_durations.append(duration_sample[mode])
        return [int(duration) for duration in real_durations]
class MMRCPSP(Instance):
    """
    Class to represent a Multi-mode Resource-Constrained Project Scheduling Problem (MMRCPSP).
    """

    def __init__(self, num_tasks, num_resources, successors, predecessors, modes, capacities, renewable):
        """
        Initialize the MMRCPSP instance.

        :param num_jobs: Number of jobs in the project.
        :param num_resources: Number of resources available.
        :param successors: List of successors for each job.
        :param predecessors: List of predecessors for each job.
        :param modes: List of modes for each job.
        :param capacities: Capacities of the resources.
        :param renewable: Boolean indicating if resources are renewable.
        """
        self.num_tasks = num_tasks
        self.num_resources = num_resources
        self.successors = successors
        self.predecessors = predecessors
        self.modes = modes
        self.capacities = capacities
        self.renewable = renewable

    def create_model(self, durations):
        """
        Create the model for the MMRCPSP.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def sample_durations(self, nb_scenarios, noise_factor=0.0):
        """
        Sample durations for the tasks in the project.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def check_feasibility(self, start_times, finish_times, *args):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_sample_length(self):
        """
        Get the length of the sample.
        This method should be implemented in subclasses.
        """
        return len(self.modes)

    def get_bounds(self, noise_factor):
        """
        Get the bounds for the durations.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def check_resource_feasibility(self, start_times, durations, demands):
        """
        Check the resource feasibility of the tasks.

        :param start_times: List of start times for each task.
        :param durations: List of durations for each task.
        :param demands: List of resource demands per task; demands[i][r] is the demand of task i for resource r.
        :return: True if feasible, False otherwise.
        """
        num_resources = self.num_resources
        num_tasks = len(durations)
        used = np.zeros((sum(durations), num_resources))

        for task in range(num_tasks):
            start_job = start_times[task]
            duration = durations[task]
            job_needs = demands[task]
            used[start_job:start_job + duration] += job_needs

        resource_feasible = True
        for t in range(sum(durations)):
            for r, resource_usage in enumerate(used[t]):
                if resource_usage > self.capacities[r]:
                    resource_feasible = False
        return resource_feasible


class MMRCPSPD(MMRCPSP):
    """
    Class to represent a Multi-mode Resource-Constrained Project Scheduling Problem with Deadlines (MMRCPSPD).
    """

    def __init__(self, num_tasks, num_resources, successors, predecessors, modes, capacities, renewable, deadlines):
        super().__init__(num_tasks, num_resources, successors, predecessors, modes, capacities, renewable)
        self.deadlines = deadlines

    def create_model(self, durations):
        model = Model()

        # resources = [model.add_renewable(capacity) for capacity in instance.capacities]
        resources = [
            model.add_renewable(capacity)
            for idx, capacity in enumerate(self.capacities)
        ]
        # We add jobs for each task and each deadline dummy task
        # jobs = [model.add_job(due_date=self.deadlines.get(idx, MAX_VALUE)) for idx in range(self.num_tasks)]
        # jobs += [
        #     model.add_job(due_date=d) for (t, d) in self.deadlines.items() # Deadline tasks should finish by deadline
        # ]
        jobs = [model.add_job() for _ in range(self.num_tasks + len(self.deadlines))]
        # Add tasks for the actual tasks and the deadlines
        tasks = [
            model.add_task(job=jobs[idx]) for idx in range(self.num_tasks + len(self.deadlines))
        ]
        for i, (t, d) in enumerate(self.deadlines.items()):
            model.add_end_before_end(tasks[t], tasks[i + self.num_tasks - 1])
            model.add_start_before_end(tasks[i + self.num_tasks - 1], tasks[0]) # Deadline tasks start at 0 to model the deadlines correctly
        # Make sure the order of durations is the same as that of modes
        for (idx, _, demands), duration in zip(self.modes, durations):
            model.add_mode(tasks[idx], resources, duration, demands)

        for idx in range(self.num_tasks + len(self.deadlines)):
            task = tasks[idx]

            for pred in self.predecessors[idx]:
                model.add_end_before_start(tasks[pred], task)

            for succ in self.successors[idx]:
                model.add_end_before_start(task, tasks[succ])
        model.set_objective(
            weight_makespan=1,
        )
        return model

    def get_bounds(self, noise_factor):
        lb = []
        ub = []
        for i, mode in enumerate(self.modes):
            duration = mode.duration
            job = mode.job
            if duration == 0:
                lb.append(0)
                ub.append(0)
            elif job >= self.num_tasks - 1:
                lb.append(duration)
                ub.append(duration)
            else:
                lower_bound = int(max(1, duration - noise_factor * np.sqrt(duration)))
                upper_bound = int(duration + noise_factor * np.sqrt(duration))
                if lower_bound == upper_bound:
                    upper_bound += 1
                lb.append(lower_bound)
                ub.append(upper_bound)
        return lb, ub
    # TODO change this to add uncertainty
    def sample_durations(self, nb_scenarios, noise_factor):
        """
        Sample durations for the tasks in the project.
        :param nb_scenarios: Number of scenarios to sample.
        :return: List of sampled durations.
        """
        lower_bound, upper_bound = self.get_bounds(noise_factor)
        duration_distributions = DiscreteUniformSampler(
            lower_bounds=lower_bound,
            upper_bounds=upper_bound
        )
        return duration_distributions.sample(nb_scenarios), duration_distributions


    def sample_mode(self, mode, noise_factor):
        """
        Sample a mode for the tasks in the project.
        :param mode: The mode to sample.
        :return: List of sampled durations.
        """
        lower_bound, upper_bound = self.get_bounds(noise_factor)
        if mode == "robust":
            durations = upper_bound
        elif mode == "mean":
            durations = [(lb + ub) // 2 for lb, ub in zip(lower_bound, upper_bound)]
        elif mode.startswith("quantile_"):
            try:
                q = float(mode.split("_")[1])
                durations = [int(lb + q * (ub - lb)) for lb, ub in zip(lower_bound, upper_bound)]
            except (IndexError, ValueError):
                raise ValueError(f"Invalid quantile format: {mode}")
        else:
            raise ValueError(f"Unknown mode type: {mode}")

        return durations


    def check_deadline_feasibility(self, finish_times):
        """
        Check the deadline feasibility of the tasks.
        :param start_times: Start times of the tasks.
        :param finish_times: Finish times of the tasks.
        :return: True if feasible, False otherwise.
        """
        for idx in range(self.num_tasks):
            if idx in self.deadlines and finish_times[idx] > self.deadlines[idx]:
                return False
        return True
    def check_feasibility(self, start_times, finish_times, durations, demands):
        """
        Check the feasibility of the solution.
        :param start_times: Start times of the tasks.
        :param finish_times: Finish times of the tasks.
        :param durations: Durations of the tasks.
        :param demands: Resource demands for each task.
        :return: True if feasible, False otherwise.
        """
        duration_feasible = self.check_duration_feasibility(start_times, finish_times, durations)
        precedence_feasible = self.check_precedence_feasibility(start_times, finish_times, self.successors)
        resource_feasible = self.check_resource_feasibility(start_times, durations, demands)
        deadline_feasible = self.check_deadline_feasibility(finish_times)
        return duration_feasible and precedence_feasible and resource_feasible and deadline_feasible
    def get_sample_length(self):
        """
        Get the length of the sample.
        :return: Length of the sample.
        """
        return len(self.modes) + len(self.deadlines)

    def get_objective_rte(self, rte_data, objective="makespan"):
        """
        Get the objective value from the RTE data.

        :param rte_data: The RTE data containing the results.
        :param objective: The type of objective to retrieve (default is "makespan").
        :return: The objective value.
        """
        if objective == "makespan":
            makespan = max([
                time for node, time in rte_data.f.items()
                if node < self.num_tasks - 1
            ])
            return makespan
        elif objective == "deadline":
            return sum(finish_time for idx, finish_time in enumerate(rte_data.f.values()) if idx in self.deadlines)
        else:
            raise ValueError("Unknown objective type.")

    def get_objective(self, schedule, objective="makespan"):
        """
        Get the objective value from the result tasks.

        :param schedule: The schedule containing the results.
        :param objective: The type of objective to retrieve (default is "makespan").
        :return: The objective value.
        """
        if objective == "makespan":
            makespan = max(task["end"] for task in schedule if task["task"] < self.num_tasks - 1)
            return makespan
        elif objective == "deadline":
            return sum(task["end"] for task in schedule if task["task"] in self.deadlines)
        else:
            raise ValueError("Unknown objective type.")

    def get_schedule(self, result_tasks):
        """
        Get the schedule for the tasks.
        """
        schedule = []
        for i, task in enumerate(result_tasks):
            if i < self.num_tasks - 1:
                schedule.append({
                    "task": i,
                    "start": task.start,
                    "end": task.end
                })
            else:
                schedule.append({
                    "task": i,
                    "start": 0,
                    "end": task.end - task.start
                })
        return schedule

    def solve_reactive(self, durations, scheduled_start_times, current_time, result_tasks, time_limit=None, initial_solution=None):
        class Mode(NamedTuple):
            job: int
            duration: int
            demands: list[int]
        # Build model with given durations
        model = Model()
        resources = [
            model.add_renewable(capacity) for capacity in self.capacities
        ]
        jobs = [
            model.add_job(due_date=self.deadlines.get(idx, MAX_VALUE))
            for idx in range(self.num_tasks)
        ]
        tasks = []
        for idx in range(self.num_tasks-1):
            scheduled_start = scheduled_start_times[idx]
            current_job = jobs[idx]
            tasks.append(model.add_task(current_job, earliest_start=scheduled_start, latest_start=scheduled_start)
                         if scheduled_start >= 0 else model.add_task(current_job, earliest_start=current_time))
        tasks.append(
            model.add_task(jobs[-1], earliest_start=scheduled_start_times[-1], latest_end=scheduled_start_times[-1] + durations[-1])
            if scheduled_start_times[-1] >= 0 else model.add_task(jobs[-1], earliest_start=current_time))
        modes = [self.modes[task.mode] for task in result_tasks]
        modes = modes[:self.num_tasks - 1] + [modes[-1]]
        modes[-1] = Mode(self.num_tasks - 1, 0, [0] * len(self.capacities))
        ds = durations[:self.num_tasks - 1] + [durations[-1]]
        for (idx, _, demands), duration in zip(modes, ds):
            model.add_mode(tasks[idx], resources, duration, demands)
        for idx in range(self.num_tasks):
            if idx == self.num_tasks - 1:
                predecessors = self.predecessors[-1]
                successors = self.successors[-1]
            else:
                predecessors = self.predecessors[idx]
                successors = self.successors[idx]
            task = tasks[idx]
            for pred in predecessors:
                try:
                    model.add_end_before_start(tasks[pred], task)
                except IndexError:
                    pass
            for succ in successors:
                try:
                    model.add_end_before_start(task, tasks[succ])
                except IndexError:
                    pass
        # TODO potentially implement the warm start solver with initial_solution
        # # Apply initial solution if provided
        # if initial_solution:
        #     for task_id, start_time in initial_solution.items():
        #         model.add_start_hint(model.tasks[task_id], start_time)
        # print all the data about the model

        # Solve model
        result = model.solve(time_limit=time_limit, display=False)
        rt = result.best.tasks
        # Extract start times and makespan
        if rt:
            start_times = [task.start for task in rt[:-1]]
            start_times += [0] * len(self.deadlines)
            start_times.append(rt[-1].start)
            finish_times = [task.end for task in rt]
            makespan = self.get_objective(self.get_schedule(rt))
            return start_times, makespan
        else:
            return None, np.inf

    def get_real_durations(self, result_tasks, duration_sample):
        """
        Get the real durations for the tasks.
        :param result_tasks: The result tasks containing the results.
        :param duration_sample: The sampled durations.
        :return: List of real durations.
        """
        real_durations = []
        for task in result_tasks:
            mode = task.mode
            real_durations.append(duration_sample[mode])
        return [int(duration) for duration in real_durations]

    def __str__(self):
        """
        String representation of the MMRCPSPD instance.
        :return: String representation.
        """
        return (f"MMRCPSPD(num_tasks={self.num_tasks}, num_resources={self.num_resources}, "
                f"successors={self.successors}, predecessors={self.predecessors}, "
                f"modes={self.modes}, capacities={self.capacities}, "
                f"renewable={self.renewable}, deadlines={self.deadlines})")

class MMRCPSPGTL(MMRCPSP):
    """
    Class to represent a Multi-mode Resource-Constrained Project Scheduling Problem with Generalized Time Lags (MMRCPSPGTL).
    """

    def __init__(self, num_tasks, num_resources, successors, predecessors, modes, capacities, renewable, start_start, start_end, end_start, end_end):
        super().__init__(num_tasks, num_resources, successors, predecessors, modes, capacities, renewable)
        self.start_start = start_start
        self.end_start = end_start
        self.start_end = start_end
        self.end_end = end_end

    def create_model(self, durations):
        model = Model()

        resources = [
            model.add_renewable(capacity)
            for idx, capacity in enumerate(self.capacities)
        ]

        jobs = [model.add_job() for _ in range(self.num_tasks)]

        tasks = [
            model.add_task(job=jobs[idx]) for idx in range(self.num_tasks)
        ]


        for (idx, _, demands), duration in zip(self.modes, durations):
            model.add_mode(tasks[idx], resources, duration, demands)

        for start_arr in self.start_start:
            model.add_start_before_start(tasks[start_arr[0]], tasks[start_arr[1]], start_arr[2])
        for end_arr in self.end_start:
            model.add_end_before_start(tasks[end_arr[0]], tasks[end_arr[1]], end_arr[2])
        # for start_arr in self.start_end:
        #     model.add_start_before_end(tasks[start_arr[0]], tasks[start_arr[1]], start_arr[2])
        # for end_arr in self.end_end:
        #     model.add_end_before_end(tasks[end_arr[0]], tasks[end_arr[1]], end_arr[2])
        model.set_objective(
            weight_makespan=1,
        )
        return model

    def get_bounds(self, noise_factor=0.0):
        lb = []
        ub = []
        for i, mode in enumerate(self.modes):
            duration = mode.duration
            job = mode.job
            if duration == 0:
                lb.append(0)
                ub.append(0)
            elif job >= self.num_tasks - 1:
                lb.append(duration)
                ub.append(duration)
            else:
                lower_bound = int(max(1, duration - noise_factor * np.sqrt(duration)))
                upper_bound = int(duration + noise_factor * np.sqrt(duration))
                if lower_bound == upper_bound:
                    upper_bound += 1
                lb.append(lower_bound)
                ub.append(upper_bound)
        return lb, ub
    # TODO change this to add uncertainty
    def sample_durations(self, nb_scenarios, noise_factor=0.0):
        """
        Sample durations for the tasks in the project.
        :param nb_scenarios: Number of scenarios to sample.
        :return: List of sampled durations.
        """
        lower_bound, upper_bound = self.get_bounds(noise_factor)
        duration_distributions = DiscreteUniformSampler(
            lower_bounds=lower_bound,
            upper_bounds=upper_bound
        )
        return duration_distributions.sample(nb_scenarios), duration_distributions


    def check_feasibility(self, start_times, finish_times, durations, demands):
        """
        Check the feasibility of the solution.
        :param start_times: Start times of the tasks.
        :param finish_times: Finish times of the tasks.
        :param durations: Durations of the tasks.
        :param demands: Resource demands for each task.
        :return: True if feasible, False otherwise.
        """
        duration_feasible = self.check_duration_feasibility(start_times, finish_times, durations)
        precedence_feasible = self.check_precedence_feasibility(start_times, finish_times, self.successors)
        resource_feasible = self.check_resource_feasibility(start_times, durations, demands)
        # Check the generalized time lags
        for start_arr in self.start_start:
            if start_times[start_arr[1]] < start_times[start_arr[0]] + start_arr[2]:
                print("fail")
                return False
        for end_arr in self.end_start:
            if start_times[end_arr[1]] < finish_times[end_arr[0]] + end_arr[2]:
                print("fail")
                return False
        # for start_arr in self.start_end:
        #     if finish_times[start_arr[1]] < start_times[start_arr[0]] + start_arr[2]:
        #         print("fail")
        #         return False
        # for end_arr in self.end_end:
        #     if finish_times[end_arr[1]] < finish_times[end_arr[0]] + end_arr[2]:
        #         print("fail")
        #         return False
        return duration_feasible and precedence_feasible and resource_feasible
    def get_sample_length(self):
        """
        Get the length of the sample.
        :return: Length of the sample.
        """
        return len(self.modes)

    def get_objective_rte(self, rte_data, objective="makespan"):
        """
        Get the objective value from the RTE data.

        :param rte_data: The RTE data containing the results.
        :param objective: The type of objective to retrieve (default is "makespan").
        :return: The objective value.
        """
        if objective == "makespan":
            makespan = max([
                time for node, time in rte_data.f.items()
                if node < self.num_tasks - 1
            ])
            return makespan
        else:
            raise ValueError("Unknown objective type.")

    def get_objective(self, schedule, objective="makespan"):
        """
        Get the objective value from the result tasks.

        :param schedule: The schedule containing the results.
        :param objective: The type of objective to retrieve (default is "makespan").
        :return: The objective value.
        """
        if objective == "makespan":
            makespan = max(task["end"] for task in schedule if task["task"] < self.num_tasks - 1)
            # print(f"---------------------------------{makespan}---------------------------------")
            return makespan
        else:
            raise ValueError("Unknown objective type.")

    def get_schedule(self, result_tasks):
        """
        Get the schedule for the tasks.
        """
        schedule = []
        for i, task in enumerate(result_tasks):
            if i < self.num_tasks - 1:
                schedule.append({
                    "task": i,
                    "start": task.start,
                    "end": task.end
                })
            else:
                schedule.append({
                    "task": i,
                    "start": 0,
                    "end": task.end - task.start
                })
        return schedule

    def solve_reactive(self, durations, scheduled_start_times, current_time, result_tasks, time_limit=None,
                       initial_solution=None):
        model = Model()
        resources = [
            model.add_renewable(capacity) for capacity in self.capacities
        ]
        jobs = [model.add_job() for _ in range(self.num_tasks)]
        tasks = []
        for idx in range(self.num_tasks - 1):
            scheduled_start = scheduled_start_times[idx]
            current_job = jobs[idx]
            tasks.append(model.add_task(current_job, earliest_start=scheduled_start, latest_start=scheduled_start)
                         if scheduled_start >= 0 else model.add_task(current_job, earliest_start=current_time))
        tasks.append(
            model.add_task(jobs[-1], earliest_start=scheduled_start_times[-1],
                           latest_end=scheduled_start_times[-1] + durations[-1])
            if scheduled_start_times[-1] >= 0 else model.add_task(jobs[-1], earliest_start=current_time))
        modes = [self.modes[task.mode] for task in result_tasks]
        ds = durations
        for (idx, _, demands), duration in zip(modes, ds):
            model.add_mode(tasks[idx], resources, duration, demands)
        for start_arr in self.start_start:
            model.add_start_before_start(tasks[start_arr[0]], tasks[start_arr[1]], start_arr[2])
        for end_arr in self.end_start:
            model.add_end_before_start(tasks[end_arr[0]], tasks[end_arr[1]], end_arr[2])
        # for start_arr in self.start_end:
        #     model.add_start_before_end(tasks[start_arr[0]], tasks[start_arr[1]], start_arr[2])
        # for end_arr in self.end_end:
        #     model.add_end_before_end(tasks[end_arr[0]], tasks[end_arr[1]], end_arr[2])
        # TODO potentially implement the warm start solver with initial_solution
        # # Apply initial solution if provided
        # if initial_solution:
        #     for task_id, start_time in initial_solution.items():
        #         model.add_start_hint(model.tasks[task_id], start_time)

        # Solve model
        result = model.solve(time_limit=time_limit, display=False, random_seed=42, num_search_workers=1)
        rt = result.best.tasks
        # Extract start times and makespan
        if rt:
            start_times = [task.start for task in rt[:-1]]
            start_times.append(rt[-1].start)
            finish_times = [task.end for task in rt]
            makespan = self.get_objective(self.get_schedule(rt))
            return start_times, makespan
        else:
            return None, np.inf

        #
        # capacities = self.capacities
        # nb_resources = len(capacities)
        # mdl = CpoModel()
        # modes = [self.modes[task.mode] for task in result_tasks]
        # demands = [modes[idx].demands for idx in range(self.num_tasks)]
        # tasks = [interval_var(name='T{}'.format(i + 1), size=durations[i]) for i in range(self.num_tasks)]
        # mdl.add(start_of(tasks[s]) + lag <= start_of(tasks[t]) for (s, t, lag) in self.start_start)
        # mdl.add(end_of(tasks[s]) + lag <= start_of(tasks[t]) for (s, t, lag) in self.end_start)
        # mdl.add(start_of(tasks[s]) + lag <= end_of(tasks[t]) for (s, t, lag) in self.start_end)
        # mdl.add(end_of(tasks[s]) + lag <= end_of(tasks[t]) for (s, t, lag) in self.end_end)
        #
        #
        # for t in range(self.num_tasks):
        #     if scheduled_start_times[t] >= 0:
        #         mdl.add(start_of(tasks[t]) == scheduled_start_times[t])
        #     else:
        #         mdl.add(start_of(tasks[t]) >= current_time)
        #
        # # Constrain capacity of needs
        # mdl.add(sum(pulse(tasks[t], demands[t][r]) for t in range(self.num_tasks) if demands[t][r] > 0) <=
        #         capacities[r] for r in range(nb_resources))
        #
        # # Add objective value
        # mdl.add(minimize(max(end_of(t) for t in tasks)))
        #
        # # Apply initial solution if provided
        # if initial_solution:
        #     starting_point = CpoModelSolution()
        #     for i in range(self.num_tasks):
        #         task_name = f'T{i}'
        #         if task_name in initial_solution:
        #             task_start = initial_solution[i]
        #             task_end = task_start + durations[i]
        #             starting_point.add_interval_var_solution(tasks[i], start=task_start, end=task_end)
        #     mdl.set_starting_point(starting_point)
        #
        # # Solve model
        # res = mdl.solve(TimeLimit=time_limit, Workers=1, LogVerbosity="Quiet")
        #
        # start_times = []
        # if res:
        #     for i in range(len(durations)):
        #         start = res.get_var_solution(tasks[i]).start
        #         start_times.append(start)
        #     makespan = res.solution.get_objective_value()
        #     return start_times, makespan
        # else:
        #     # for (s, t, lag) in self.end_end:
        #     #     e_s = scheduled_start_times[s] + durations[s] if scheduled_start_times[s] >= 0 else 'unknown'
        #     #     e_t = scheduled_start_times[t] + durations[t] if scheduled_start_times[t] >= 0 else 'unknown'
        #     #     if e_s == 'unknown' or e_t == 'unknown':
        #     #         continue
        #     #     print(
        #     #         f"original duration of task {s}: {self.modes[result_tasks[s].mode].duration}, "
        #     #         f"original duration of task {t}: {self.modes[result_tasks[t].mode].duration}")
        #     #     print(f"duration of task {s}: {durations[s]}, duration of task {t}: {durations[t]}")
        #     #     print(f"EE constraint: end({s}) + {lag} <= end({t}) | end_s: {e_s}, end_t: {e_t}")
        #     return None, np.inf





    def sample_mode(self, mode, noise_factor):
        """
        Sample a mode for the tasks in the project.
        :param mode: The mode to sample.
        :param noise_factor: The noise factor to apply to the bounds.
        :return: List of sampled durations.
        """
        lower_bound, upper_bound = self.get_bounds(noise_factor)
        if mode == "robust":
            durations = upper_bound
        elif mode == "mean":
            durations = [(lb + ub) // 2 for lb, ub in zip(lower_bound, upper_bound)]
        elif mode == "quantile_0.25":
            durations = [int(lb + 0.25 * (ub - lb)) for lb, ub in zip(lower_bound, upper_bound)]
        elif mode == "quantile_0.75":
            durations = [int(lb + 0.75 * (ub - lb)) for lb, ub in zip(lower_bound, upper_bound)]
        elif mode == "quantile_0.9":
            durations = [int(lb + 0.9 * (ub - lb)) for lb, ub in zip(lower_bound, upper_bound)]
        else:
            raise ValueError("Unknown mode type.")
        return durations

    def get_real_durations(self, result_tasks, duration_sample):
        """
        Get the real durations for the tasks.
        :param result_tasks: The result tasks containing the results.
        :param duration_sample: The sampled durations.
        :return: List of real durations.
        """
        real_durations = []
        # # Get the modes of the tasks
        # result_tasks_mode = [task.mode for task in result_tasks]
        # print(result_tasks_mode)
        # print(duration_sample)
        for task in result_tasks:
            mode = task.mode
            real_durations.append(duration_sample[mode])
        # print([int(duration) for duration in real_durations])
        return [int(duration) for duration in real_durations]

    def get_deterministic_makespan(self):
        """
        Get the deterministic makespan for the tasks.
        :return: makespan of the schedule.
        """
        model = self.create_model([task.duration for task in self.modes])
        result = model.solve(time_limit=50000, display=False, random_seed=42, num_search_workers=1)
        rt = result.best.tasks
        if rt:
            # print(rt)
            print(f"Makespan: {result.best.makespan}")
#TODO implement the other problem instances