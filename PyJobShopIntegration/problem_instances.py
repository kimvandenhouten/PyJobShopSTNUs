from typing import NamedTuple

import numpy as np
from pyjobshop import Model, MAX_VALUE

from PyJobShopIntegration.Sampler import DiscreteUniformSampler


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
                    return False
        return True

    def check_feasibility(self, start_times, finish_times, *args):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_sample_length(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_bounds(self):
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

    def get_bounds(self):
        """
        Get the bounds for the durations.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def check_resource_feasibility(self, start_times, durations, demands):
        """
        Check the resource feasibility of the tasks.
        :param start_times: Start times of the tasks.
        :param durations: Durations of the tasks.
        :param demands: Resource demands for each task.
        :return: True if feasible, False otherwise.
        """
        num_resources = len(self.capacities)
        num_jobs = len(durations)
        used = np.zeros((sum(durations), num_resources))

        for job in range(num_jobs):
            start_job = start_times[job]
            duration = durations[job]
            job_needs = demands[job]
            used[start_job:start_job + duration] += job_needs
        # np.set_printoptions(threshold=np.inf)
        # print("Used resources: ", used)
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
        class Mode(NamedTuple):
            job: int
            duration: int
            demands: list[int]

            def __str__(self):
                return f"Mode(job={self.job}, duration={self.duration}, demands={self.demands})"
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


    def sample_mode(self, mode):
        """
        Sample a mode for the tasks in the project.
        :param mode: The mode to sample.
        :return: List of sampled durations.
        """
        lower_bound, upper_bound = self.get_bounds()
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
            print(f"---------------------------------{makespan}---------------------------------")
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

    def solve_reactive(self, durations, scheduled_start_times, current_time, time_limit=None, initial_solution=None):

        # Build model with given durations
        print("Durations: ", durations)
        model = self.create_model(durations)

        # Apply fixed start times or release times based on current schedule
        for task_id, scheduled_start in enumerate(scheduled_start_times):
            task = model.tasks[task_id]
            if scheduled_start >= 0:
                model.tasks[task_id] = model.add_task(model.jobs[task.job], earliest_start=scheduled_start, latest_start=scheduled_start)
                # model.tasks[task_id].latest_start = scheduled_start
                # model.tasks[task_id].earliest_start = scheduled_start
            else:
                model.tasks[task_id] = model.add_task(model.jobs[task.job], earliest_start=current_time)
        print("Tasks: ", model.tasks)
        # TODO potentially implement the warm start solver with initial_solution
        # # Apply initial solution if provided
        # if initial_solution:
        #     for task_id, start_time in initial_solution.items():
        #         model.add_start_hint(model.tasks[task_id], start_time)

        # Solve model
        result = model.solve(time_limit=time_limit)
        result_tasks = result.best.tasks

        # Extract start times and makespan
        if result_tasks:
            start_times = [task.start for task in result_tasks]
            finish_times = [task.end for task in result_tasks]
            makespan = self.get_objective(self.get_schedule(result_tasks))
            return start_times, makespan
        else:
            return None, np.inf

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

    def __init__(self, num_tasks, num_resources, successors, predecessors, modes, capacities, renewable, args):
        super().__init__(num_tasks, num_resources, successors, predecessors, modes, capacities, renewable)
        # TODO implement the gtl arguments
        self.args = args

    def create_model(self, durations):
        pass

    def sample_durations(self, nb_scenarios, noise_factor=0.0):
        pass

#TODO implement the other problem instances