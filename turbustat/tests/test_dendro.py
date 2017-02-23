# Licensed under an MIT open source license - see LICENSE


'''
Tests for Dendrogram statistics
'''

import numpy as np
import numpy.testing as npt

from ..statistics import Dendrogram_Stats, DendroDistance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


min_deltas = np.logspace(-1.5, 0.5, 40)


def test_DendroStat():

    tester = Dendrogram_Stats(dataset1["cube"],
                              min_deltas=min_deltas)
    tester.run(periodic_bounds=False)

    npt.assert_allclose(tester.numfeatures,
                        computed_data["dendrogram_val"])


def test_DendroDistance():

    tester_dist = \
        DendroDistance(dataset1["cube"],
                       dataset2["cube"],
                       min_deltas=min_deltas,
                       periodic_bounds=False).distance_metric()

    npt.assert_almost_equal(tester_dist.histogram_distance,
                            computed_distances["dendrohist_distance"])
    npt.assert_almost_equal(tester_dist.num_distance,
                            computed_distances["dendronum_distance"])
