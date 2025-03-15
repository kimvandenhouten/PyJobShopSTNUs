import os

import general.logger
from temporal_networks.cstnu_tool.call_java_cstnu_tool import CSTNUTool, DCAlgorithm
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU
logger = general.logger.get_logger(__name__)


def main():
    file_list = [
        ("rte_error_minimal_example.stnu", True)
    ]

    for (file_name, expected_dc) in file_list:
        instance_location = os.path.abspath(f"temporal_networks/cstnu_tool/xml_files/{file_name}")
        if not os.path.exists(instance_location):
            logger.warning(f"warning: could not find {instance_location}")
            continue
        logger.debug(f"running CSTNUTool on {file_name}")

        output_location = instance_location.replace(".stnu", "-output.stnu")

        found_dc = CSTNUTool.run_dc_alg(instance_location, output_location, DCAlgorithm.FD_STNU_IMPROVED)

        logger.debug(f'found dc is {found_dc}')
        if found_dc != expected_dc:
            logger.warning(f'WARNING: Network was unexpectedly found {"" if found_dc else "not "} to be DC')

        # Run with Python RTE*
        dc_stnu = STNU.from_graphml(output_location)
        rte_data = rte_star(dc_stnu)
        logger.debug(f'Python schedule: {rte_data.f}')
        for key in rte_data.f:
            print(f'{dc_stnu.translation_dict[key]} -> {rte_data.f[key]}')

        # Run with Java RTE*
        schedule = CSTNUTool.run_rte(output_location)
        if schedule:

            logger.debug(f"Java parsed schedule: {schedule}")
        else:
            logger.debug("could not parse schedule")


if __name__ == "__main__":
    main()
