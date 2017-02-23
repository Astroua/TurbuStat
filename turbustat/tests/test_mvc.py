# Licensed under an MIT open source license - see LICENSE


'''
Test functions for MVC
'''

import numpy.testing as npt

from ..statistics import MVC, MVC_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_MVC_method():
    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester.run()

    npt.assert_allclose(tester.ps1D, computed_data['mvc_val'])
    npt.assert_almost_equal(tester.slope, computed_data['mvc_slope'])


def test_MVC_distance():
    tester_dist = \
        MVC_Distance(dataset1, dataset2).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['mvc_distance'])
