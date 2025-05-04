import time
import copy
import numpy as np
import general.logger
from rcpsp_max.solvers.check_feasibility import check_feasibility_rcpsp_max

logger = general.logger.get_logger(__name__)

class ProactiveScheduler:
    """
    proactive scheduler supporting both RCPSP_MAX and Flexible Job Shop models.

    The provided model must implement:
      - get_bound(mode): returns lower or upper bound durations
      - solve(durations, time_limit, mode): returns (res, data) with data['start']
      - sample_durations(n): returns list of duration samples
      - solve_saa(samples, time_limit): returns (res, start_times)

    mode: one of ["robust", "quantile_0.25", "quantile_0.5", "quantile_0.75", "quantile_0.9", "SAA", "SAA_smart"]
    """
    def __init__(self, scheduling_model, time_limit=60, mode="robust", nb_scenarios_saa=10):
        """
        scheduling_model: RCPSP_MAX or PyJobShop model wrapped to expose required methods
        """
        self.model = scheduling_model
        self.time_limit = time_limit
        self.mode = mode
        self.nb_scenarios_saa = nb_scenarios_saa

        # Results storage
        self.start_times = None
        self.estimated_durations = None
        self.data_dict = {
            "instance_folder": getattr(scheduling_model, 'instance_folder', None),
            "instance_id": getattr(scheduling_model, 'instance_id', None),
            "noise_factor": getattr(scheduling_model, 'noise_factor', None),
            "method": f"proactive_{mode}",
            "time_limit": time_limit,
            "feasibility": False,
            "obj": np.inf,
            "time_offline": None,
            "time_online": None,
            "start_times": None,
            "real_durations": None,
            "estimated_durations": None,
        }

    @classmethod
    def from_rcpsp_model(cls, rcpsp_model, **kwargs):
        """Initialize scheduler from RCPSP_MAX model"""
        return cls(rcpsp_model, **kwargs)

    @classmethod
    def from_fjsp_model(cls, fjsp_model, **kwargs):
        """Initialize scheduler from Flexible Job Shop model"""
        # fjsp_model should implement required interface
        return cls(fjsp_model, **kwargs)

    def _get_bounds(self):
        lb = self.model.get_bound(mode="lower_bound")
        ub = self.model.get_bound(mode="upper_bound")
        return lb, ub

    def _compute_quantile(self, lb, ub, p):
        if lb == ub:
            return lb
        return [int(lb[i] + p * (ub[i] - lb[i] + 1) - 1) for i in range(len(lb))]

    def build_schedule(self):
        """Offline schedule construction"""
        global res
        start_offline = time.time()
        lb, ub = self._get_bounds()

        if self.mode == "robust":
            durations = ub
        elif self.mode.startswith("quantile"):
            p = float(self.mode.split("_")[1])
            durations = self._compute_quantile(lb, ub, p)
        elif self.mode == "SAA":
            durations = None
        elif self.mode == "SAA_smart":
            durations = None
        else:
            raise NotImplementedError(f"Unknown mode {self.mode}")

        self.estimated_durations = durations
        self.data_dict["estimated_durations"] = durations

        if self.mode in ["robust"] + [m for m in ["quantile_0.25", "quantile_0.5", "quantile_0.75", "quantile_0.9"]]:
            logger.debug(f"Solving schedule with durations {durations}")
            res, data = self.model.solve(durations, time_limit=self.time_limit, mode="Quiet")
            if res:
                self.start_times = data['start'].tolist()
                self.data_dict["start_times"] = self.start_times
        elif self.mode in ["SAA", "SAA_smart"]:
            samples = []
            if self.mode == "SAA":
                samples = self.model.sample_durations(self.nb_scenarios_saa)
                res, self.start_times = self.model.solve_saa(samples, self.time_limit)
            else:
                lb, ub = lb, ub
                for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    samples.append(self._compute_quantile(lb, ub, quantile))
                res, self.start_times = self.model.solve_saa(samples, self.time_limit)
            if res:
                self.data_dict["start_times"] = self.start_times

        self.data_dict["time_offline"] = time.time() - start_offline
        self.data_dict["estimated_start_times"] = self.start_times
        return res

    def evaluate_feasibility(self, duration_sample):
        """Online feasibility check"""
        data_dict = copy.deepcopy(self.data_dict)
        data_dict["real_durations"] = duration_sample
        if self.start_times is None:
            return data_dict

        start_online = time.time()
        finish_times = [self.start_times[i] + duration_sample[i] for i in range(len(self.start_times))]
        feasible = check_feasibility_rcpsp_max(
            self.start_times,
            finish_times,
            duration_sample,
            self.model.capacity,
            self.model.needs,
            self.model.temporal_constraints,
        )
        data_dict["time_online"] = time.time() - start_online
        data_dict["feasibility"] = feasible
        data_dict["obj"] = max(finish_times) if feasible else np.inf
        return data_dict

    def run(self, duration_sample=None):
        """
        Combine offline build and optional online evaluation.
        If duration_sample provided, run evaluate_feasibility.
        Returns data_dict or list of data_dict if duration_sample provided.
        """
        self.build_schedule()
        if duration_sample is not None:
            return self.evaluate_feasibility(duration_sample)
        return self.data_dict
def check_feasibility_fjsp(start_times, durations, precedence_relations, machine_assignments):
    """
    Check the feasibility of a Flexible Job Shop schedule based on:
    - start_times: list of start times for tasks
    - durations: list of durations for tasks
    - precedence_relations: list of (predecessor_idx, successor_idx) task pairs
    - machine_assignments: dict mapping machine_idx -> list of task indices assigned to it (ordered by schedule)

    Returns: True if schedule is feasible, False otherwise
    """
    finish_times = [start_times[i] + durations[i] for i in range(len(start_times))]
    for pred, succ in precedence_relations:
        if finish_times[pred] > start_times[succ]:
            return False

    for machine, tasks in machine_assignments.items():
        for i in range(len(tasks) - 1):
            t1 = tasks[i]
            t2 = tasks[i + 1]
            if finish_times[t1] > start_times[t2]:
                return False

    return True
