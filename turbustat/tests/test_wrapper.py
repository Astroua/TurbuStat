# Licensed under an MIT open source license - see LICENSE

from ..statistics import stats_wrapper
from ._testing_data import \
    dataset1, dataset2


def test_wrapper():

    run_wrapper = stats_wrapper(dataset1, dataset2)
