# Licensed under an MIT open source license - see LICENSE


'''
Test functions for SCF
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import SCF, SCF_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testSCF(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_SCF_method(self):
        self.tester = SCF(dataset1["cube"][0])
        self.tester.run()

        assert np.allclose(self.tester.scf_surface, computed_data['scf_val'])

    def test_SCF_distance(self):
        self.tester_dist = \
            SCF_Distance(dataset1["cube"][0],
                         dataset2["cube"][0]).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['scf_distance'])
