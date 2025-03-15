import general.logger
import numpy as np
from temporal_networks.stnu import STNU
from pyjobshop.Model import Model, Solution
from pyjobshop.Model import StartBeforeEnd, StartBeforeStart, EndBeforeEnd, EndBeforeStart, SetupTime
from PyJobShopIntegration.utils import find_schedule_per_resource
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
logger = general.logger.get_logger(__name__)


class PyJobShopSTNU(STNU):
    def __init__(self, origin_horizon=True):
        super().__init__(origin_horizon)

    @classmethod
    def from_concrete_model(cls, model: Model):
        stnu = cls(origin_horizon=False)
        for task_idx, task in enumerate(model.tasks):
            task_start = stnu.add_node(f'{task_idx}_{STNU.EVENT_START}')
            task_finish = stnu.add_node(f'{task_idx}_{STNU.EVENT_FINISH}')

            # TODO: add contingent links now with dummy data but this should come from a data source
            logger.warning(f'Contingent link for task {task_idx} set with bound (2, 7) but in an'
                           f'correct implementation this should be read from problem data')
            stnu.add_contingent_link(task_start, task_finish, 2, 7)

        for cons in model.constraints.end_before_start:
            stnu.add_end_before_start_constraints(cons)

        for cons in model.constraints.end_before_end:
            stnu.add_end_before_end_constraints(cons)

        for cons in model.constraints.start_before_end:
            stnu.add_start_before_end_constraints(cons)

        for cons in model.constraints.start_before_start:
            stnu.add_start_before_start_constraints(cons)

        for cons in model.constraints.setup_times:
            stnu.add_setup_times(cons)

        return stnu

    def add_end_before_end_constraints(self, cons: EndBeforeEnd):
        """
        e_1 + d \\leq e_2 is in the STNU translated to e_2 --(-delay)--> e_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_FINISH}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_FINISH}']

        # TODO: check if delay is correctly set in STNU
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

    def add_end_before_start_constraints(self, cons: EndBeforeStart):
        """
        e_1 + d \\leq s_2 is in the STNU translated to s_2 --(-delay)--> e_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_FINISH}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_START}']

        # TODO: check if delay is correctly set in STNU
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

    def add_start_before_end_constraints(self, cons: StartBeforeEnd):
        """
        s_1 + d \\leq e_2 is in the STNU translated to e_2 --(-delay)--> s_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_START}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_FINISH}']

        # TODO: check if delay is correctly set in STNU
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

    def add_start_before_start_constraints(self, cons: StartBeforeStart):
        """
        s_1 + d \\leq s_2 is in the STNU translated to s_2 --(-delay)--> s_1.
        """
        pred_idx = self.translation_dict_reversed[f'{cons.task1}_{STNU.EVENT_START}']
        suc_idx = self.translation_dict_reversed[f'{cons.task2}_{STNU.EVENT_START}']

        # TODO: check if delay is correctly set in STNU
        self.set_ordinary_edge(suc_idx, pred_idx, -cons.delay)

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
                    f"{first_idx}_{STNU.EVENT_FINISH}"]  # Get translation index from finish of predecessor
                suc_idx_start = self.translation_dict_reversed[
                    f"{second_idx}_{STNU.EVENT_START}"]  # Get translation index from start of successor

                # add constraint between predecessor and successor
                self.set_ordinary_edge(suc_idx_start, pred_idx_finish, 0)

        # TODO: implement set-up times
        if len(model.constraints.setup_times) > 0:
            raise NotImplementedError(f'Setup times are not yet implemented')