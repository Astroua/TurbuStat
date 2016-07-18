# Licensed under an MIT open source license - see LICENSE

'''
Test functions for VCS
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import VCS, VCS_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testVCS(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_VCS_method(self):
        self.tester = VCS(dataset1["cube"]).run()

        npt.assert_allclose(self.tester.ps1D, computed_data['vcs_val'])

    def test_VCS_distance(self):
        self.tester_dist = \
            VCS_Distance(dataset1["cube"], dataset2["cube"])
        self.tester_dist = self.tester_dist.distance_metric()

        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['vcs_distance'])
