# Licensed under an MIT open source license - see LICENSE

import pytest
import numpy as np

from ..statistics import stats_wrapper, statistics_list
from ._testing_data import \
    dataset1, dataset2

spacers = np.arange(2, len(statistics_list) + 1, 2)


# Split these into smaller tests to avoid timeout errors on Travis
@pytest.mark.parametrize(('stats'),
                         [statistics_list[i - 2:i] for i in
                          spacers])
def test_wrapper(stats):

    stats_wrapper(dataset1, dataset2,
                  statistics=stats)
