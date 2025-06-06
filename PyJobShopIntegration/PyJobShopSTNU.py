import general.logger
import numpy as np
from temporal_networks.stnu import STNU
from pyjobshop.Model import Model, Solution
from pyjobshop.Model import StartBeforeEnd, StartBeforeStart, EndBeforeEnd, EndBeforeStart, SetupTime
from PyJobShopIntegration.utils import find_schedule_per_resource
from PyJobShopIntegration.Sampler import DiscreteRVSampler
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm

logger = general.logger.get_logger(__name__)


class PyJobShopSTNU(STNU):
    def __init__(self, origin_horizon=True):
        super().__init__(origin_horizon)

    @classmethod
    def from_concrete_model(cls, model: Model, duration_distributions: DiscreteRVSampler, multimode=False, result_tasks=None):
        stnu = cls(origin_horizon=False)

        # prepare storage for exactly the nodes we will sample
        stnu._contingent_nodes = []
        lower_bounds, upper_bounds = duration_distributions.get_bounds()
        # Only add the bounds for the selected modes
        if multimode:
            lower_bounds = np.array([lower_bounds[task.mode] for task in result_tasks])
            upper_bounds = np.array([upper_bounds[task.mode] for task in result_tasks])
        for task_idx, task in enumerate(model.tasks):
            # task_start = stnu.add_node(f'{task_idx}_{STNU.EVENT_START}')
            # task_finish = stnu.add_node(f'{task_idx}_{STNU.EVENT_FINISH}')
            # if lower_bounds[task_idx] == upper_bounds[task_idx]:
            #     stnu.add_tight_constraint(task_start, task_finish, lower_bounds[task_idx])
            s = stnu.add_node(f'{task_idx}_{STNU.EVENT_START}')
            f = stnu.add_node(f'{task_idx}_{STNU.EVENT_FINISH}')

            lb, ub = lower_bounds[task_idx], upper_bounds[task_idx]
            if lb == ub:
                stnu.add_tight_constraint(s, f, lb)
            else:
                stnu.add_contingent_link(s, f, lb, ub)
                # remember to sample for this finish node
                stnu._contingent_nodes.append(f)

        # then your existing temporal constraints…
        for cons in model.constraints.end_before_start:
            stnu.add_end_before_start_constraints(cons)
        for cons in model.constraints.end_before_end:
            stnu.add_end_before_end_constraints(cons)
        for cons in model.constraints.start_before_end:
            stnu.add_start_before_end_constraints(cons)
        for cons in model.constraints.start_before_start:
            stnu.add_start_before_start_constraints(cons)
        return stnu

    def add_end_before_end_constraints(self, cons: EndBeforeEnd):
        """
        e_1 + d \\leq e_2 is in the STNU translated to e_2 --(-delay)--> e_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_FINISH}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_FINISH}']
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

    def add_end_before_start_constraints(self, cons: EndBeforeStart):
        """
        e_1 + d \\leq s_2 is in the STNU translated to s_2 --(-delay)--> e_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_FINISH}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_START}']
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

    def add_start_before_end_constraints(self, cons: StartBeforeEnd):
        """
        s_1 + d \\leq e_2 is in the STNU translated to e_2 --(-delay)--> s_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_START}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_FINISH}']
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

    def add_start_before_start_constraints(self, cons: StartBeforeStart):
        """
        s_1 + d \\leq s_2 is in the STNU translated to s_2 --(-delay)--> s_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_START}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_START}']
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

    def add_resource_chains(self, sol: Solution, model: Model):
        schedule_per_resource = find_schedule_per_resource(sol)

        # Add up set-up delays
        setup_delay = {
            (c.machine, c.task1, c.task2): c.duration
            for c in model.constraints.setup_times
        }

        # Add resource chains
        for machine, sequence in schedule_per_resource.items():
            for i in range(len(sequence) - 1):
                first_idx = sequence[i]
                second_idx = sequence[i + 1]
                logger.info(f'Add resource chain between task {first_idx} and task {second_idx}')
                # the finish of the predecessor should precede the start of the successor
                pred_idx_finish = self.translation_dict_reversed[
                    f"{first_idx}_{STNU.EVENT_FINISH}"]  # Get translation index from finish of predecessor
                suc_idx_start = self.translation_dict_reversed[
                    f"{second_idx}_{STNU.EVENT_START}"]  # Get translation index from start of successor

                # If there is a sequence-dependent set-up time, determine delay
                delay = setup_delay.get((machine, first_idx, second_idx), 0)
                # add constraint between predecessor and successor, with sequence dependent set up times this is a delay
                self.set_ordinary_edge(suc_idx_start, pred_idx_finish, -delay)

                # ---------- NEW: make start(succ) executable, controlled by finish(pred) ----------
                # self.node_types[suc_idx_start] = self.EXECUTABLE_TP
                # self.set_ordinary_edge(suc_idx_start, pred_idx_finish, 0)  # start ≤ finish
                # self.set_ordinary_edge(pred_idx_finish, suc_idx_start, 0)  # finish ≤ start

                if delay > 0:
                    logger.info(f"Resource {machine}: SDST {first_idx} → {second_idx} delay={delay}")
