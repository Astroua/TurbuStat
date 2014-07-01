
'''
Test functions for MVC
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import MVC, MVC_distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testMVC(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_MVC_method(self):
        self.tester = MVC(dataset1["centroid"][0] * dataset1["centroid_error"][0] ** 2.,
                            dataset1["moment0"][0] * dataset1["moment0_error"][0] ** 2.,
                            dataset1["linewidth"][0] * dataset1["linewidth_error"][0] ** 2.,
                            dataset1["centroid"][1])
        self.tester.run()

        assert np.allclose(self.tester.ps1D, computed_data['mvc_val'])

    def test_MVC_distance(self):
        self.tester_dist = \
            MVC_distance(dataset1, dataset2).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['mvc_distance'])
