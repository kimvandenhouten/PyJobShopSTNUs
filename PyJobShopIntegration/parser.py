def parse_data(file, problem_type):
    if problem_type == "rcpsp":
        return parse_data_rcpsp(file)
    elif problem_type == "fjsp":
        return parse_data_fjsp(file)

# TODO implement parser for rcpsp instances
def parse_data_rcpsp(file):
    pass
# TODO implement parser for fjsp instances
def parse_data_fjsp(file):
    pass