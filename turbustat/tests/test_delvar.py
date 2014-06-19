
'''
Test functions for Delta Variance
'''

import numpy as np

from ..statistics import DeltaVariance, DeltaVariance_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testDelVar():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances

    def test_DelVar_method(self):
        self.tester = DeltaVariance_Distance(dataset1["integrated_intensity"], dataset1["integrated_intensity_error"][0], dataset2["integrated_intensity"], dataset2["integrated_intensity_error"][0])
        assert np.allclose(self.tester.delvar1.delta_var, self.computed_data['delvar_val'])

    def test_DelVar_distance(self):
        assert np.allclose(self.tester.distance_metric().distance, self.computed_distances['delvar_distance'])