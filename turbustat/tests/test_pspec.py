
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
        self.tester = PSpec_Distance(dataset1, dataset2).pspec1.ps1D
        assert np.allclose(self.tester, self.computed_data['pspec_val'])

    def test_PSpec_distance(self):
    	self.tester = PSpec_Distance(dataset1, dataset2).distance_metric().distance
    	assert np.allclose(self.tester, self.computed_distances['pspec_distance'])