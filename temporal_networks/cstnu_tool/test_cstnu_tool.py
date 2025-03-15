import os

import general.logger
from temporal_networks.cstnu_tool.call_java_cstnu_tool import CSTNUTool

logger = general.logger.get_logger(__name__)


def test_cstnu_tool():
    file_name, expected_dc = ("example_rcpsp_max_stnu.stnu", True)
    instance_location = os.path.abspath(f"temporal_networks/cstnu_tool/xml_files/{file_name}")

    if not os.path.exists(instance_location):
        logger.warning(f"warning: could not find {instance_location}")

    logger.debug(f"running CSTNUTool on {file_name}")

    found_dc = CSTNUTool.run_dc_alg(instance_location)
    if found_dc != expected_dc:
        logger.warning(f'WARNING: Network was unexpectedly found {"" if found_dc else "not "} to be DC')

    schedule = CSTNUTool.run_rte(instance_location)
    if schedule:
        logger.debug(f"parsed schedule: {schedule}")
    else:
        logger.debug("could not parse schedule")
