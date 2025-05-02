from pyjobshop import Model, MAX_VALUE

class MMRCPSP():
    """
    Class to represent a Multi-mode Resource-Constrained Project Scheduling Problem (MMRCPSP).
    """

    def __init__(self, num_jobs, num_resources, successors, predecessors, modes, capacities, renewable):
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
        self.num_jobs = num_jobs
        self.num_resources = num_resources
        self.successors = successors
        self.predecessors = predecessors
        self.modes = modes
        self.capacities = capacities
        self.renewable = renewable

    def create_model(self):
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

class MMRCPSPD(MMRCPSP):
    """
    Class to represent a Multi-mode Resource-Constrained Project Scheduling Problem with Deadlines (MMRCPSPD).
    """

    def __init__(self, num_jobs, num_resources, successors, predecessors, modes, capacities, renewable, deadlines):
        super().__init__(num_jobs, num_resources, successors, predecessors, modes, capacities, renewable)
        self.deadlines = deadlines

    def create_model(self):
        model = Model()

        # It's not necessary to define jobs, but it will add coloring to the plot.
        jobs = [model.add_job() for _ in range(self.num_jobs)]
        tasks = [
            model.add_task(job=jobs[idx],
                           latest_end=self.deadlines[idx] if idx in self.deadlines else MAX_VALUE)
            for idx in range(self.num_jobs)
        ]
        # resources = [model.add_renewable(capacity) for capacity in instance.capacities]
        resources = [
            model.add_renewable(capacity) if self.renewable[idx] else model.add_non_renewable(capacity)
            for idx, capacity in enumerate(self.capacities)
        ]
        for idx, duration, demands in self.modes:
            model.add_mode(tasks[idx], resources, duration, demands)

        for idx in range(self.num_jobs):
            task = tasks[idx]

            for pred in self.predecessors[idx]:
                model.add_end_before_start(tasks[pred], task)

            for succ in self.successors[idx]:
                model.add_end_before_start(task, tasks[succ])
        return model
    def sample_durations(self, nb_scenarios):
        pass

class MMRCPSPGTL(MMRCPSP):
    """
    Class to represent a Multi-mode Resource-Constrained Project Scheduling Problem with Generalized Time Lags (MMRCPSPGTL).
    """

    def __init__(self, num_jobs, num_resources, successors, predecessors, modes, capacities, renewable, args):
        super().__init__(num_jobs, num_resources, successors, predecessors, modes, capacities, renewable)
        # TODO implement the gtl arguments
        self.args = args

    def create_model(self):
        pass

    def sample_durations(self, nb_scenarios):
        pass

#TODO implement the other problem instances