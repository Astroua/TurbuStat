# Licensed under an MIT open source license - see LICENSE


'''
Test function for Wavelet
'''

import numpy as np
import numpy.testing as npt

from ..statistics import Wavelet, Wavelet_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_Wavelet_method():
    tester = Wavelet(dataset1["moment0"])
    tester.run()
    npt.assert_almost_equal(tester.values, computed_data['wavelet_val'])


def test_Wavelet_distance():
    tester_dist = \
        Wavelet_Distance(dataset1["moment0"],
                         dataset2["moment0"]).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['wavelet_distance'])
