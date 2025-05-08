import os
import numpy as np
from pyjobshop.Model import Model
from temporal_networks.stnu import STNU

from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.reactive_left_shift import group_shift_solution_resequenced
from PyJobShopIntegration.utils import find_schedule_per_resource
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from PyJobShopIntegration.simulator import Simulator
from PyJobShopIntegration.plot_gantt_and_stats import plot_simulation_statistics
from pyjobshop.plot import plot_machine_gantt
import general.logger

logger = general.logger.get_logger(__name__)

class InstanceParser:
    """
    Parses raw data definitions into structured jobs, tasks, deadlines.
    """
    def __init__(self, num_machines: int, data: list[list[list[tuple[int,int]]]],
                 job_deadlines: dict[int,int]):
        self.num_machines = num_machines
        self.raw_data = data
        self.job_deadlines = job_deadlines

    def parse(self):
        # validate lengths and return structured
        return self.num_machines, self.raw_data, self.job_deadlines

class CPModelBuilder:
    """
    Builds and solves the PyJobShop CP model.
    """
    def __init__(self, num_machines: int, data: list, job_deadlines: dict):
        self.num_machines = num_machines
        self.data = data
        self.job_deadlines = job_deadlines
        self.model = Model()
        self.tasks = {}

    def build(self):
        # set objective, machines, tasks, constraints, deadlines
        machines = [self.model.add_machine(name=f"M{i}") for i in range(self.num_machines)]
        deadline_res = self.model.add_renewable(capacity=999, name="DeadlineRes")

        for j, job_data in enumerate(self.data):
            job = self.model.add_job(name=f"Job{j}", due_date=self.job_deadlines[j])
            for t_idx, options in enumerate(job_data):
                task = self.model.add_task(job=job, name=f"T({j},{t_idx})")
                self.tasks[(j,t_idx)] = task
                for dur, m in options:
                    self.model.add_mode(task, machines[m], dur)
            # precedences and dummy deadline tasks
            for t in range(len(job_data)-1):
                self.model.add_end_before_start(self.tasks[(j,t)], self.tasks[(j,t+1)])
            last = self.tasks[(j,len(job_data)-1)]
            dt = self.model.add_task(name=f"Deadline{j}", earliest_start=0, latest_end=self.job_deadlines[j])
            self.model.add_mode(dt, deadline_res, 1)
            self.model.add_end_before_start(last, dt)

        return self.model, self.tasks

    def solve(self):
        res = self.model.solve(display=True)
        sol = res.best
        sol = group_shift_solution_resequenced(sol, self.model)
        return sol, res.objective

class STNUBuilder:
    """
    Converts CP solution into an STNU with resource chains and deadlines.
    """
    def __init__(self, model: Model, solution, duration_sampler: DiscreteUniformSampler):
        self.model = model
        self.solution = solution
        self.sampler = duration_sampler

    def build(self):
        stnu = PyJobShopSTNU.from_concrete_model(self.model, self.sampler)
        stnu.add_resource_chains(self.solution, self.model)
        return stnu

    def add_deadlines(self, job_deadlines: dict):
        # assume tasks map known externally
        for j, dl in job_deadlines.items():
            # locate finish node and add deadline constraint
            pass

class DCChecker:
    """
    Writes STNU to XML and runs the Java DC algorithm.
    """
    def __init__(self, stnu: STNU, name: str, xml_dir: str = "xml_files"):
        self.stnu = stnu
        self.name = name
        self.xml_dir = xml_dir
        os.makedirs(xml_dir, exist_ok=True)

    def check(self):
        stnu_to_xml(self.stnu, self.name, self.xml_dir)
        dc, out_loc = run_dc_algorithm(self.xml_dir, self.name)
        return dc, out_loc

class Pipeline:
    """
    Orchestrates the full offline + DC check + online simulation + plotting.
    """
    def __init__(self, num_machines, data, job_deadlines):
        self.parser = InstanceParser(num_machines, data, job_deadlines)
        self.builder = None
        self.stnu_builder = None
        self.dc_checker = None

    def run(self, runs=1000):
        nm, data, deadlines = self.parser.parse()
        self.builder = CPModelBuilder(nm, data, deadlines)
        model, tasks = self.builder.build()
        sol, obj = self.builder.solve()
        sampler = DiscreteUniformSampler(
            lower_bounds=np.ones(len(model.tasks), dtype=int),
            upper_bounds=np.full(len(model.tasks), 10, dtype=int)
        )
        self.stnu_builder = STNUBuilder(model, sol, sampler)
        stnu = self.stnu_builder.build()

        self.dc_checker = DCChecker(stnu, "fjsp")
        dc, graphml = self.dc_checker.check()
        if not dc:
            logger.warning("Not controllable")
            return

        sim = Simulator(model, stnu, sol, sampler)
        summary = sim.run_many(runs)

        plot_machine_gantt(summary['first_solution'], model.data(), plot_labels=True)
        plot_simulation_statistics(summary['makespans'], summary['violations'], summary['total_runs'])

# Usage example (in main script)
# pipeline = Pipeline(NUM_MACHINES, data, job_deadlines)
# pipeline.run(runs=1000)
