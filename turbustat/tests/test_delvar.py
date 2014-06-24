
'''
Test functions for Delta Variance
'''

import numpy as np

from ..statistics import DeltaVariance, DeltaVariance_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testDelVar():

    def __init__(self):
        print "Setting up class"
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances

    def test_DelVar_method(self):
        print "testing method"
        self.tester = DeltaVariance(dataset1["integrated_intensity"][0], dataset1["integrated_intensity"][1], dataset1["integrated_intensity_error"][0], diam_ratio=1.5, lags=None)
        self.tester.run()
        assert np.allclose(self.tester.delta_var, self.computed_data['delvar_val'])

    def test_DelVar_distance(self):
        print "testing distance"
        self.tester_dist = DeltaVariance_Distance(dataset1["integrated_intensity"], dataset1["integrated_intensity_error"][0], dataset2["integrated_intensity"], dataset2["integrated_intensity_error"][0], fiducial_model = self.tester)
        assert np.allclose(self.tester_dist.distance_metric().distance, self.computed_distances['delvar_distance'])