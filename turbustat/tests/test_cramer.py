
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

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_cramer(self):
        self.tester = Cramer_Distance(dataset1["cube"][0], dataset2["cube"][0])
        self.tester.distance_metric()
        assert np.allclose(self.tester.data_matrix1,
                           computed_data["cramer_val"])
        npt.assert_almost_equal(self.tester.distance,
                                computed_distances['cramer_distance'])
