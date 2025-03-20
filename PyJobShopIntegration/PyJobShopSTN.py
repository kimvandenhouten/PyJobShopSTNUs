import copy

import general.logger
import numpy as np
from temporal_networks.stn import STN
from pyjobshop.Model import Model, Solution
from pyjobshop.Model import StartBeforeEnd, StartBeforeStart, EndBeforeEnd, EndBeforeStart, SetupTime
from PyJobShopIntegration.utils import find_schedule_per_resource
from PyJobShopIntegration.Sampler import DiscreteRVSampler
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm

logger = general.logger.get_logger(__name__)


class PyJobShopSTN(STN):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_concrete_model(cls, model: Model, sample_durations: np.ndarray):
        stn = cls()

        for task_idx, task in enumerate(model.tasks):
            task_start = stn.add_node(f'{task_idx}_{STN.EVENT_START}')
            task_finish = stn.add_node(f'{task_idx}_{STN.EVENT_FINISH}')

            stn.add_tight_constraint(task_start, task_finish, sample_durations[task_idx])

        for cons in model.constraints.end_before_start:
            stn.add_end_before_start_constraints(cons)

        for cons in model.constraints.end_before_end:
            stn.add_end_before_end_constraints(cons)

        for cons in model.constraints.start_before_end:
            stn.add_start_before_end_constraints(cons)

        for cons in model.constraints.start_before_start:
            stn.add_start_before_start_constraints(cons)

        for cons in model.constraints.setup_times:
            stn.add_setup_times(cons)

        return stn

    def add_end_before_end_constraints(self, cons: EndBeforeEnd):
        """
        e_1 + d \\leq e_2 is in the STNU translated to e_2 --(-delay)--> e_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STN.EVENT_FINISH}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STN.EVENT_FINISH}']
        self.set_edge(suc_idx, pred_idx, -cons.delay)

    def add_end_before_start_constraints(self, cons: EndBeforeStart):
        """
        e_1 + d \\leq s_2 is in the STNU translated to s_2 --(-delay)--> e_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STN.EVENT_FINISH}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STN.EVENT_START}']
        self.set_edge(suc_idx, pred_idx, -cons.delay)

    def add_start_before_end_constraints(self, cons: StartBeforeEnd):
        """
        s_1 + d \\leq e_2 is in the STNU translated to e_2 --(-delay)--> s_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STN.EVENT_START}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STN.EVENT_FINISH}']
        self.set_edge(suc_idx, pred_idx, -cons.delay)

    def add_start_before_start_constraints(self, cons: StartBeforeStart):
        """
        s_1 + d \\leq s_2 is in the STNU translated to s_2 --(-delay)--> s_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STN.EVENT_START}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STN.EVENT_START}']
        self.set_edge(suc_idx, pred_idx, -cons.delay)

    def add_setup_times(self, cons: SetupTime):
        """
        Set-up times are machine-dependent, so they can only been added when the schedule per resource is known
        """
        raise NotImplementedError

    def add_resource_chains(self, sol: Solution, model: Model):
        schedule_per_resource = find_schedule_per_resource(sol)

        # Add resource chains
        for machine, sequence in schedule_per_resource.items():
            for i in range(len(sequence) - 1):
                first_idx = sequence[i]
                second_idx = sequence[i + 1]
                logger.info(f'Add resource chain between task {first_idx} and task {second_idx}')
                # the finish of the predecessor should precede the start of the successor
                pred_idx_finish = self.translation_dict_reversed[
                    f"{first_idx}_{STN.EVENT_FINISH}"]  # Get translation index from finish of predecessor
                suc_idx_start = self.translation_dict_reversed[
                    f"{second_idx}_{STN.EVENT_START}"]  # Get translation index from start of successor

                # add constraint between predecessor and successor
                self.set_edge(suc_idx_start, pred_idx_finish, 0)

        # TODO: implement set-up times
        if len(model.constraints.setup_times) > 0:
            raise NotImplementedError(f'Setup times are not yet implemented')

    def roll_out_schedule(self, solution):
        simulated_solution = copy.deepcopy(solution)
        self.floyd_warshall()
        makespan = - self.shortest_distances[STN.HORIZON_IDX][STN.ORIGIN_IDX]
        for i in range(len(solution.tasks)):
            start_node = self.translation_dict_reversed[f'{i}_start']
            finish_node = self.translation_dict_reversed[f'{i}_finish']

            start_time = -self.shortest_distances[start_node][STN.ORIGIN_IDX]
            finish_time = -self.shortest_distances[finish_node][STN.ORIGIN_IDX]

            simulated_solution.tasks[i].start = start_time
            simulated_solution.tasks[i].end = finish_time
        return simulated_solution, makespan

