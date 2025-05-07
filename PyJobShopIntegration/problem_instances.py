from typing import NamedTuple

import numpy as np
from pyjobshop import Model, MAX_VALUE

from PyJobShopIntegration.Sampler import DiscreteUniformSampler


# Parent class of all instances, could include more important methods if needed
class Instance():
    def get_objective(self, rte_data, objective="makespan"):
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
            model.add_renewable(capacity) if self.renewable[idx] else model.add_non_renewable(capacity)
            for idx, capacity in enumerate(self.capacities)
        ]
        # We add jobs for each task and each deadline dummy task
        jobs = [model.add_job(due_date=self.deadlines.get(idx, MAX_VALUE)) for idx in range(self.num_tasks)]
        jobs += [
            model.add_job(due_date=d) for (t, d) in self.deadlines.items() # Deadline tasks should finish by deadline
        ]
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
        # TODO implement sampling logic
        lower_bound, upper_bound = self.get_bounds(noise_factor)
        duration_distributions = DiscreteUniformSampler(
            lower_bounds=lower_bound,
            upper_bounds=upper_bound
        )
        return duration_distributions.sample(nb_scenarios), duration_distributions

    # TODO potentially need more checks
    def check_feasibility(self, start_times, finish_times, *args):
        """
        Check the feasibility of the solution.
        :param start_times: Start times of the tasks.
        :param finish_times: Finish times of the tasks.
        :return: True if feasible, False otherwise.
        """
        for idx in range(self.num_tasks):
            if idx in self.deadlines and finish_times[idx] > self.deadlines[idx]:
                return False
        return True

    def get_sample_length(self):
        """
        Get the length of the sample.
        :return: Length of the sample.
        """
        return len(self.modes) + len(self.deadlines)

    def get_objective(self, rte_data, objective="makespan"):
        """
        Get the objective value from the RTE data.

        :param rte_data: The RTE data containing the results.
        :param objective: The type of objective to retrieve (default is "makespan").
        :return: The objective value.
        """
        if objective == "makespan":
            makespan = max([
                time for node, time in rte_data.f.items()
                if node < self.num_tasks
            ])
            return makespan
        elif objective == "deadline":
            return sum(finish_time for idx, finish_time in enumerate(rte_data.f.values()) if idx in self.deadlines)
        else:
            raise ValueError("Unknown objective type.")

    def get_schedule(self, result_tasks):
        """
        Get the schedule for the tasks.
        """
        schedule = []
        for i, task in enumerate(result_tasks):
            if i < self.num_tasks:
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