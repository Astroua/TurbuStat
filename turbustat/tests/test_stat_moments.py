# Licensed under an MIT open source license - see LICENSE


'''
Test functions for Kurtosis
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import StatMoments, StatMomentsDistance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class TestMoments(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_moments(self):
        self.tester = StatMoments(dataset1["integrated_intensity"][0], 5)
        self.tester.run()
        assert np.allclose(self.tester.kurtosis_hist[1],
                           computed_data['kurtosis_val'])
        assert np.allclose(self.tester.skewness_hist[1],
                           computed_data['skewness_val'])

    def test_moment_distance(self):
        self.tester_dist = \
            StatMomentsDistance(dataset1["integrated_intensity"][0],
                                dataset2["integrated_intensity"][0], 5)
        self.tester_dist.distance_metric()
        npt.assert_almost_equal(self.tester_dist.kurtosis_distance,
                                computed_distances['kurtosis_distance'])
        npt.assert_almost_equal(self.tester_dist.skewness_distance,
                                computed_distances['skewness_distance'])
