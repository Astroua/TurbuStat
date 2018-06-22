# Licensed under an MIT open source license - see LICENSE


def assert_between(value, lower, upper):
    '''
    Check if a value is between two values.
    '''

    within_lower = value >= lower
    within_upper = value <= upper

    if within_lower and within_upper:
        return
    else:
        raise AssertionError("{0} not within {1} and {2}".format(value, lower,
                                                                 upper))
