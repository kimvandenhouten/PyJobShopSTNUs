import logging
from temporal_networks.rte_star import rte_star
from PyJobShopIntegration.utils import sample_for_rte, rte_data_to_pyjobshop_solution

logger = logging.getLogger(__name__)

class Simulator:
    def __init__(self, model, stnu, solution, sampler, objective="makespan"):
        """
        :param model: PyJobShop Model instance
        :param stnu: STNU object used for simulation (must match sample_for_rte)
        :param solution: Deterministic PyJobShop Solution to base dispatching on
        :param sampler: A duration sampler implementing .sample() -> np.ndarray
        :param objective: Objective for evaluation ("makespan", etc.)
        """
        self.model = model
        # keep original STNU intact; we'll deep-copy for each run
        self._template_stnu = stnu
        self.solution = solution
        self.sampler = sampler
        self.objective = objective
        self.num_tasks = len(model.tasks)

    def run_once(self):
        """
        Perform one RTE* execution on a fresh STNU copy:
         - draw durations
         - deep-copy template STNU
         - map to contingent links
         - run rte_star
         - if infeasible (no rte_data.f), return (None, None)
         - otherwise return (simulated_solution, objective_value)
        """
        import copy
        durations = self.sampler.sample()
        logger.debug(f"Sampled durations: {durations}")

        # reset STNU to original state
        stnu = copy.deepcopy(self._template_stnu)

        sample = sample_for_rte(durations, stnu)
        rte_data = rte_star(stnu, oracle="sample", sample=sample)

        if not hasattr(rte_data, 'f'):
            logger.warning("[RTE*] Infeasible execution sample; skipping this run.")
            return None, None

        simulated_solution, objective_value = rte_data_to_pyjobshop_solution(
            self.solution,
            stnu,
            rte_data,
            self.num_tasks,
            self.objective
        )
        return simulated_solution, objective_value

    def run_many(self, runs=100):
        """
        Run multiple simulations, skipping infeasible draws.
        Returns summary dict:
          - makespans: list of objective values
          - violations: count of deadline misses
          - total_runs: attempted runs
          - simulated_runs: count of successful simulations
          - first_solution: first successful simulated solution
        """
        makespans = []
        violations = 0
        simulated_count = 0
        first_solution = None

        for i in range(runs):
            sim_sol, obj = self.run_once()
            if sim_sol is None:
                continue
            simulated_count += 1
            makespans.append(obj)
            if first_solution is None:
                first_solution = sim_sol

            # count deadline violations
            for idx, task in enumerate(sim_sol.tasks):
                latest = self.model.tasks[idx].latest_end
                if latest is not None and task.end > latest:
                    violations += 1
                    break

        return {
            'makespans': makespans,
            'violations': violations,
            'total_runs': runs,
            'simulated_runs': simulated_count,
            'first_solution': first_solution
        }
