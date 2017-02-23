# Licensed under an MIT open source license - see LICENSE

'''
Test functions for VCA
'''

import numpy.testing as npt

from ..statistics import VCA, VCA_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_VCA_method():
    tester = VCA(dataset1["cube"])
    tester.run()
    npt.assert_allclose(tester.ps1D, computed_data['vca_val'])
    npt.assert_almost_equal(tester.slope, computed_data['vca_slope'],
                            decimal=3)


def test_VCA_distance():
    tester_dist = \
        VCA_Distance(dataset1["cube"],
                     dataset2["cube"]).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['vca_distance'])
