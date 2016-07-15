# Licensed under an MIT open source license - see LICENSE


'''
Test functions for Delta Variance
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import DeltaVariance, DeltaVariance_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testDelVar(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_DelVar_method(self):
        self.tester = \
            DeltaVariance(dataset1["moment0"],
                          weights=dataset1["moment0_error"][0])
        self.tester.run()
        npt.assert_allclose(self.tester.delta_var, computed_data['delvar_val'])

    def test_DelVar_distance(self):
        self.tester_dist = \
            DeltaVariance_Distance(dataset1["moment0"],
                                   dataset2["moment0"],
                                   weights1=dataset1["moment0_error"][0],
                                   weights2=dataset2["moment0_error"][0])
        self.tester_dist.distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['delvar_distance'],
                                decimal=3)
