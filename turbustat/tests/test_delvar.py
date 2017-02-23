# Licensed under an MIT open source license - see LICENSE


'''
Test functions for Delta Variance
'''

import numpy.testing as npt

from ..statistics import DeltaVariance, DeltaVariance_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_DelVar_method():
    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1["moment0_error"][0])
    tester.run()
    npt.assert_allclose(tester.delta_var, computed_data['delvar_val'])
    npt.assert_almost_equal(tester.slope, computed_data['delvar_slope'])


def test_DelVar_distance():
    tester_dist = \
        DeltaVariance_Distance(dataset1["moment0"],
                               dataset2["moment0"],
                               weights1=dataset1["moment0_error"][0],
                               weights2=dataset2["moment0_error"][0])
    tester_dist.distance_metric()
    npt.assert_almost_equal(tester_dist.curve_distance,
                            computed_distances['delvar_curve_distance'],
                            decimal=3)
    npt.assert_almost_equal(tester_dist.slope_distance,
                            computed_distances['delvar_slope_distance'],
                            decimal=3)
