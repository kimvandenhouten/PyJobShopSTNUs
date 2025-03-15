from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.dc_checking import convert_to_normal_form, determine_dc
from temporal_networks.stnu import STNU

# Example Hunsberger slide 118 (controllable)
name = "presentation"
stnu = STNU(origin_horizon=False)
a_start = stnu.add_node('a_start')
a_finish = stnu.add_node('a_finish')
b_start = stnu.add_node('b_start')
b_finish = stnu.add_node('b_finish')
c_start = stnu.add_node('c_start')
c_finish = stnu.add_node('c_finish')
d_start = stnu.add_node('d_start')
d_finish = stnu.add_node('d_finish')
e_start = stnu.add_node('e_start')
e_finish = stnu.add_node('e_finish')

# Set ordinary processing times
stnu.set_ordinary_edge(a_start, a_finish, 2)
stnu.set_ordinary_edge(a_finish, a_start, -2)

stnu.set_ordinary_edge(b_start, b_finish, 5)
stnu.set_ordinary_edge(b_finish, b_start, -5)

stnu.set_ordinary_edge(c_start, c_finish, 3)
stnu.set_ordinary_edge(c_finish, c_start, -3)

stnu.set_ordinary_edge(e_start, e_finish, 1)
stnu.set_ordinary_edge(e_finish, e_start, -1)

# Set contingent processeing times
stnu.add_contingent_link(d_start, d_finish, 1, 2)

# Set resource chains
resource_chains = [("d", "a"), ("a", "e"), ("e", "c"), ("a", "b")]
#resource_chains = [("a", "d"), ("d", "e"), ("d", "b"), ("e", "c")]

for (pred, suc) in resource_chains:
    suc_node = stnu.translation_dict_reversed[f"{suc}_start"]
    pred_node = stnu.translation_dict_reversed[f"{pred}_finish"]
    stnu.set_ordinary_edge(suc_node, pred_node, 0)

# Set precedence constraints
precedence_constraints = ("d", "e", 3), ("e", "d", -3), ("c", "a", -6), ("b", "c", 1), ("a", "b", 2)

for (pred, suc, weight) in precedence_constraints:
    suc_node = stnu.translation_dict_reversed[f"{suc}_start"]
    pred_node = stnu.translation_dict_reversed[f"{pred}_start"]
    stnu.set_ordinary_edge(suc_node, pred_node, -weight)
stnu_to_xml(stnu, f"example_presentation", "temporal_networks/cstnu_tool/xml_files")
#dc, estnu = determine_dc(stnu, dispatchability=True)
#if dc:
 #   stnu_to_xml(estnu, f"output_example_rcpsp_max", "temporal_networks/cstnu_tool/xml_files")


