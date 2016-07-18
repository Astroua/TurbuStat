# Licensed under an MIT open source license - see LICENSE


'''
Test functions for MVC
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import MVC, MVC_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testMVC(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_MVC_method(self):
        self.tester = MVC(dataset1["centroid"],
                          dataset1["moment0"],
                          dataset1["linewidth"],
                          dataset1["centroid"][1])
        self.tester.run()

        print self.tester.ps1D
        print computed_data['mvc_val']

        npt.assert_allclose(self.tester.ps1D, computed_data['mvc_val'])

    def test_MVC_distance(self):
        self.tester_dist = \
            MVC_Distance(dataset1, dataset2).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['mvc_distance'])
