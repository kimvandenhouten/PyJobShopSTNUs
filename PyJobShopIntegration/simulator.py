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

        if not hasattr(rte_data, "f"):
            logger.warning("RTE* returned infeasible result for sample. Skipping.")
            return None, None

        simulated_solution, objective = rte_data_to_pyjobshop_solution(
            self.solution, self.stnu, rte_data, self.num_tasks, self.objective
        )
        return simulated_solution, objective

    def run_many(self, runs=100):
        makespans = []
        violations = 0
        all_solutions = []
        failed_runs = 0

        for run in range(runs):
            sim_solution, obj = self.run_once()

            if sim_solution is None:
                failed_runs += 1
                continue  # skip this failed run

            all_solutions.append(sim_solution)
            makespans.append(obj)

            if any(
                    task.end > self.model.tasks[i].latest_end
                    for i, task in enumerate(sim_solution.tasks)
                    if self.model.tasks[i].latest_end is not None
            ):
                violations += 1

        summary = {
            "makespans": makespans,
            "violations": violations,
            "total_runs": runs - failed_runs,
            "failed_runs": failed_runs,
            "first_solution": all_solutions[0] if all_solutions else None,
            "all_solutions": all_solutions,
        }

        logger.info(f"[SIMULATION] Completed {runs - failed_runs}/{runs} successful runs.")
        logger.info(f"[SIMULATION] {failed_runs} runs failed due to RTE* infeasibility.")

        return summary

