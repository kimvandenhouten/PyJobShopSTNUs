# Import
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
import numpy as np
from rcpsp_max.temporal_networks.stnu_rcpsp_max import RCPSP_STNU, get_resource_chains, add_resource_chains
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
import os
import general.logger
from temporal_networks.cstnu_tool.call_java_cstnu_tool import CSTNUTool
logger = general.logger.get_logger(__name__)

wrong_instances = []
# SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'
time_limit_cp_stnu = 60
noise_factor = 1
xml_name = "example_rte_error"
xml_directory = "temporal_networks/cstnu_tool/xml_files"

# PREPARE DATA AND DURATION SAMPLES
for instance_folder in ["j30"]:
    for instance_id in [30]:
        logger.info(f'start instance {instance_folder} {instance_id}')
        rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor)
        np.random.seed(SEED)
        test_durations_samples = rcpsp_max.sample_durations(1)


        # Build STNU and check DC
        durations = rcpsp_max.get_bound(mode="upper_bound")
        res, schedule = rcpsp_max.solve(durations, time_limit=time_limit_cp_stnu)

        if res:
            # Build the STNU using the instance information and the resource chains
            schedule = schedule.to_dict('records')
            resource_chains, resource_assignments = get_resource_chains(schedule, rcpsp_max.capacity, rcpsp_max.needs,
                                                                        complete=True)
            stnu = RCPSP_STNU.from_rcpsp_max_instance(rcpsp_max.durations, rcpsp_max.temporal_constraints,
                                                      noise_factor=rcpsp_max.noise_factor)
            stnu = add_resource_chains(stnu, resource_chains)
            stnu_to_xml(stnu, f"{xml_name}", xml_directory)


            file_list = [
                (f"{xml_name}.stnu", True)
            ]


            for (file_name, expected_dc) in file_list:
                instance_location = os.path.abspath(f"{xml_directory}/{file_name}")
                if not os.path.exists(instance_location):
                    logger.warning(f"warning: could not find {instance_location}")
                    continue
                logger.debug(f"running CSTNUTool on {file_name}")

                output_location = instance_location.replace(".stnu", "-output.stnu")

                found_dc = CSTNUTool.run_dc_alg(instance_location, output_location)

                logger.debug(f'found dc is {found_dc}')
                if found_dc != expected_dc:
                    logger.warning(f'WARNING: Network was unexpectedly found {"" if found_dc else "not "} to be DC')

                schedule = CSTNUTool.run_rte(output_location)
                if schedule:
                    logger.debug(f"parsed schedule: {schedule}")

                    # Check that all starts are before finish
                    for i in range(rcpsp_max.num_tasks-2):
                        start = schedule[f"{i+1}_start"]
                        finish = schedule[f"{i+1}_finish"]
                        if finish < start:
                            logger.debug(f'finish ({finish}) < start ({start} for task {i+1} ')
                            wrong_instances.append((instance_folder, instance_id))
                else:
                    logger.debug("could not parse schedule")


                logger.info(f'wrong instances {wrong_instances}')


