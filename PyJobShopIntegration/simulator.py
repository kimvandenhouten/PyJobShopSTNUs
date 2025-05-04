import numpy as np
import logging
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU
from PyJobShopIntegration.utils import sample_for_rte, rte_data_to_pyjobshop_solution

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, model, stnu, solution, sampler, objective="makespan"):
        """
        :param model: PyJobShop Model instance
        :param stnu: STNU object (with resource + temporal constraints encoded)
        :param solution: Deterministic solution to base dispatching on
        :param sampler: A duration sampler implementing .sample()
        :param objective: "makespan", "earliness", etc.
        """
        self.model = model
        self.stnu = stnu
        self.solution = solution
        self.sampler = sampler
        self.objective = objective
        self.num_tasks = len(model.tasks)

    def run_once(self):
        durations = self.sampler.sample()
        logger.debug(f"Sampled durations: {durations}")

        sample = sample_for_rte(durations, self.stnu)
        rte_data = rte_star(self.stnu, oracle="sample", sample=sample)

        simulated_solution, objective = rte_data_to_pyjobshop_solution(
            self.solution, self.stnu, rte_data, self.num_tasks, self.objective
        )
        return simulated_solution, objective

    def run_many(self, runs=100):
        makespans = []
        violations = 0
        all_solutions = []

        for run in range(runs):
            sim_solution, obj = self.run_once()
            all_solutions.append(sim_solution)
            makespans.append(obj)

            if any(task.end > self.model.tasks[i].latest_end
                   for i, task in enumerate(sim_solution.tasks)
                   if self.model.tasks[i].latest_end is not None):
                violations += 1

        summary = {
            "makespans": makespans,
            "violations": violations,
            "total_runs": runs,
            "first_solution": all_solutions[0] if runs > 0 else None,
            "all_solutions": all_solutions
        }
        return summary
