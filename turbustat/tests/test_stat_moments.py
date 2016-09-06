# Licensed under an MIT open source license - see LICENSE


'''
Test functions for Kurtosis
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import StatMoments, StatMoments_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class TestMoments(TestCase):

    def test_moments(self):
        self.tester = StatMoments(dataset1["moment0"])
        self.tester.run()

        # This simply ensures the data set will run.
        # There are subtle differences due to matching the bins
        # between the sets. So all tests are completed below

    def test_moment_distance(self):
        self.tester_dist = \
            StatMoments_Distance(dataset1["moment0"],
                                 dataset2["moment0"])
        self.tester_dist.distance_metric()

        assert np.allclose(self.tester_dist.moments1.kurtosis_hist[1],
                           computed_data['kurtosis_val'])
        assert np.allclose(self.tester_dist.moments1.skewness_hist[1],
                           computed_data['skewness_val'])

        npt.assert_almost_equal(self.tester_dist.kurtosis_distance,
                                computed_distances['kurtosis_distance'])
        npt.assert_almost_equal(self.tester_dist.skewness_distance,
                                computed_distances['skewness_distance'])
