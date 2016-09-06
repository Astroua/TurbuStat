# Licensed under an MIT open source license - see LICENSE


'''
Test functions for Cramer
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import Cramer_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testCramer(TestCase):

    def test_cramer(self):
        self.tester = Cramer_Distance(dataset1["cube"],
                                      dataset2["cube"]).distance_metric()

        npt.assert_allclose(self.tester.data_matrix1,
                            computed_data["cramer_val"])
        npt.assert_almost_equal(self.tester.distance,
                                computed_distances['cramer_distance'])

    def test_cramer_spatial_diff(self):

        small_data = dataset1["cube"][0][:, :26, :26]

        self.tester2 = Cramer_Distance(small_data, dataset2["cube"])
        self.tester2.distance_metric()
        self.tester3 = Cramer_Distance(dataset2["cube"], small_data)
        self.tester3.distance_metric()

        npt.assert_almost_equal(self.tester2.distance, self.tester3.distance)
