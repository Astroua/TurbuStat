# Licensed under an MIT open source license - see LICENSE


'''
Test function for Wavelet
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import Wavelet, Wavelet_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testWavelet(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_Wavelet_method(self):
        self.tester = Wavelet(dataset1["moment0"])
        self.tester.run()
        assert np.allclose(self.tester.values, computed_data['wavelet_val'])

    def test_Wavelet_distance(self):
        self.tester_dist = \
            Wavelet_Distance(dataset1["moment0"],
                             dataset2["moment0"]).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['wavelet_distance'])
