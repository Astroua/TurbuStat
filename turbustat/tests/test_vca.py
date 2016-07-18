# Licensed under an MIT open source license - see LICENSE

'''
Test functions for VCA
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import VCA, VCA_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testVCA(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_VCA_method(self):
        self.tester = VCA(dataset1["cube"])
        self.tester.run()
        npt.assert_allclose(self.tester.ps1D, computed_data['vca_val'])

    def test_VCA_distance(self):
        self.tester_dist = \
            VCA_Distance(dataset1["cube"],
                         dataset2["cube"]).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['vca_distance'])
