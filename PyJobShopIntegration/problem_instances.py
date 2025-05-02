from pyjobshop import Model, MAX_VALUE
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

    def sample_durations(self, nb_scenarios):
        """
        Sample durations for the tasks in the project.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def check_feasibility(self, start_times, finish_times, *args):
        raise NotImplementedError("Subclasses should implement this method.")

class MMRCPSPD(MMRCPSP):
    """
    Class to represent a Multi-mode Resource-Constrained Project Scheduling Problem with Deadlines (MMRCPSPD).
    """

    def __init__(self, num_tasks, num_resources, successors, predecessors, modes, capacities, renewable, deadlines):
        super().__init__(num_tasks, num_resources, successors, predecessors, modes, capacities, renewable)
        self.deadlines = deadlines

    def create_model(self, durations):
        model = Model()

        # It's not necessary to define jobs, but it will add coloring to the plot.
        jobs = [model.add_job() for _ in range(self.num_tasks)]
        tasks = [
            model.add_task(job=jobs[idx],
                           latest_end=self.deadlines[idx] if idx in self.deadlines else MAX_VALUE)
            for idx in range(self.num_tasks)
        ]
        # resources = [model.add_renewable(capacity) for capacity in instance.capacities]
        resources = [
            model.add_renewable(capacity) if self.renewable[idx] else model.add_non_renewable(capacity)
            for idx, capacity in enumerate(self.capacities)
        ]
        # Make sure the order of durations is the same as that of modes
        for (idx, _, demands), duration in zip(self.modes, durations):
            model.add_mode(tasks[idx], resources, duration, demands)

        for idx in range(self.num_tasks):
            task = tasks[idx]

            for pred in self.predecessors[idx]:
                model.add_end_before_start(tasks[pred], task)

            for succ in self.successors[idx]:
                model.add_end_before_start(task, tasks[succ])
        return model
    def sample_durations(self, nb_scenarios):
        pass

    # TODO potentially need more checks
    def check_feasibility(self, start_times, finish_times, *args):
        """
        Check the feasibility of the solution.
        :param start_times: Start times of the tasks.
        :param finish_times: Finish times of the tasks.
        :return: True if feasible, False otherwise.
        """
        for idx in range(self.num_tasks):
            if finish_times[idx] > self.deadlines[idx]:
                return False
        return True

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

    def sample_durations(self, nb_scenarios):
        pass

#TODO implement the other problem instances