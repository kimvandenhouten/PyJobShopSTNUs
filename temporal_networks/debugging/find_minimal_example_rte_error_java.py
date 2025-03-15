# Import
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
import numpy as np
from rcpsp_max.temporal_networks.stnu_rcpsp_max import RCPSP_STNU, get_resource_chains, add_resource_chains
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
import os
import general.logger
from temporal_networks.cstnu_tool.call_java_cstnu_tool import CSTNUTool
logger = general.logger.get_logger(__name__)
from temporal_networks.stnu import STNU, SampleStrategy

from copy import deepcopy
from typing import Any

def rcpsp_to_stnu(instance_folder, instance_id, noise_factor):

    logger.info(f'start instance {instance_folder} {instance_id}')
    rcpsp_max = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor)
    np.random.seed(SEED)
    test_durations_samples = rcpsp_max.sample_durations(1)

    # Build STNU and check DC
    durations = rcpsp_max.get_bound(mode="upper_bound")
    res, schedule = rcpsp_max.solve(durations, time_limit=time_limit_cp_stnu)

    if not res:
        return None

    # Build the STNU using the instance information and the resource chains
    schedule = schedule.to_dict('records')
    resource_chains, resource_assignments = get_resource_chains(schedule, rcpsp_max.capacity,
                                                                rcpsp_max.needs,
                                                                complete=True)
    stnu = RCPSP_STNU.from_rcpsp_max_instance(rcpsp_max.durations, rcpsp_max.temporal_constraints,
                                              noise_factor=rcpsp_max.noise_factor)
    stnu = add_resource_chains(stnu, resource_chains)
    return stnu


def check_rte_problem():
    stnu_to_xml(stnu, f"{xml_name}", xml_directory)
    file_name = f"{xml_name}.stnu"
    instance_location = os.path.abspath(f"{xml_directory}/{file_name}")
    if not os.path.exists(instance_location):
        logger.warning(f"warning: could not find {instance_location}")

    logger.debug(f"running CSTNUTool on {file_name}")

    output_location = instance_location.replace(".stnu", "-output.stnu")

    found_dc = CSTNUTool.run_dc_alg(instance_location, output_location)

    logger.debug(f'found dc is {found_dc}')

    if found_dc is False:
        return False

    schedule = CSTNUTool.run_rte(output_location)
    if schedule:
        logger.info(f"parsed schedule: {schedule}")
        return False

    else:
        logger.info("could not parse schedule")
        return True

# SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'
time_limit_cp_stnu = 60
noise_factor = 1
xml_name = "example_rte_error"
xml_directory = "temporal_networks/cstnu_tool/xml_files"
instance_folder = "j30"
instance_id = 30

# Find minimal example
BUILD_FROM_RCPSP = True
if BUILD_FROM_RCPSP:
    stnu = rcpsp_to_stnu(instance_folder, instance_id, noise_factor)
else:
    stnu = RCPSP_STNU.from_graphml(f"{xml_directory}/{xml_name}")
if not stnu:
    print(f'The model has no solution for instance id {instance_id}')
    raise Exception("infeasible rcpsp instance")

problem_persists = check_rte_problem()

logger.info(f'Problem persists: {problem_persists}')

# TODO: iterative remove edges / nodes to check whether problem persists
# TODO: check whether sampling is fixed

removed_edges: list[Any] = []
removed_nodes: set[Any] = set()

for node in stnu.edges:
    for edge in stnu.edges[node].values():
        if edge.uc_label:
            e2 = stnu.edges[edge.node_to][edge.node_from]
            assert e2 is not None and e2.lc_label
        elif edge.lc_label:
            e2 = stnu.edges[edge.node_to][edge.node_from]
            assert e2 is not None and e2.uc_label


def find_edge_to_remove() -> bool:
    # for e, t in removed_edges:
    #    stnu2.remove_edge(e.node_from, e.node_to, t)
    found = False

    for node in stnu.edges:
        for edge in list(stnu.edges[node].values()):
            e2 = deepcopy(edge)
            for type in [STNU.ORDINARY_LABEL, STNU.LC_LABEL]:
                if stnu.remove_edge(edge.node_from, edge.node_to, type):
                    if type == STNU.LC_LABEL:
                        e3 = deepcopy(stnu.edges[edge.node_to][edge.node_from])
                        assert e3 is not None and e3.uc_label
                        removed = stnu.remove_edge(e3.node_from, e3.node_to, STNU.UC_LABEL)
                        assert removed
                    else:
                        e3 = None

                    try:
                        problem_persists = check_rte_problem()
                    except:
                        problem_persists = False

                    if problem_persists:
                        found = True
                        removed_edges.append((e2, type))
                        if e3 is not None:
                            assert type == STNU.LC_LABEL
                            removed_edges.append((e3, STNU.UC_LABEL))
                        else:
                            assert type == STNU.ORDINARY_LABEL

                        print(f'can remove edge {edge.node_from}, {edge.node_to}, {type}')
                    else:
                        # re-add the edge
                        if type == STNU.ORDINARY_LABEL:
                            assert e3 is None
                            stnu.set_ordinary_edge(e2.node_from, e2.node_to, e2.weight)
                        else:
                            assert type == STNU.LC_LABEL
                            assert (e3 is not None
                                    and e3.node_from == e2.node_to
                                    and e2.node_from == e3.node_to)
                            stnu.set_labeled_edge(e2.node_from, e2.node_to,
                                                  e2.lc_weight, e2.lc_label,
                                                  STNU.LC_LABEL)
                            stnu.set_labeled_edge(e3.node_from, e3.node_to,
                                                  e3.uc_weight, e3.uc_label,
                                                  STNU.UC_LABEL)
    return found


while True:
    if not find_edge_to_remove():
        break
for node in stnu.edges:
    if len(stnu.edges[node]) == 0:
        if all(node not in edge_set for
               edge_set in stnu.edges.values()):
            removed_nodes.add(node)
for node in removed_nodes:
    stnu.remove_node(node)

stnu_to_xml(stnu, f"rte_error_minimal_example", "temporal_networks/cstnu_tool/xml_files")

print(f"{len(removed_edges)} edges can be removed: "
      f"{[(e.node_from, e.node_to, t) for (e, t) in removed_edges]}")
print(f"{len(removed_nodes)} nodes can be removed: {removed_nodes}")