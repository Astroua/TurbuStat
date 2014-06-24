
'''
Test functions for PSpec
'''

import numpy as np

from ..statistics import PowerSpectrum, PSpec_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testPSpec():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_PSpec_method(self):
<<<<<<< HEAD
        self.tester = PowerSpectrum(dataset1["integrated_intensity"][0] * dataset1["integrated_intensity_error"][0] ** 2., dataset1["integrated_intensity"][1])
        assert np.allclose(self.tester.ps1D, self.computed_data['pspec_val'])

    def test_PSpec_distance(self):
    	self.tester_dist = PSpec_Distance(dataset1, dataset2, fiducial_model = self.tester).distance_metric().distance
    	assert np.allclose(self.tester_dist, self.computed_distances['pspec_distance'])
=======
        self.tester = PSpec_Distance(dataset1, dataset2).pspec1.ps1D
        assert np.allclose(self.tester, self.computed_data['pspec_val'])

    def test_PSpec_distance(self):
    	self.tester = PSpec_Distance(dataset1, dataset2).distance_metric().distance
    	assert np.allclose(self.tester, self.computed_distances['pspec_distance'])
>>>>>>> db672c6e73207e459a27d399616f5fb1caf04199
