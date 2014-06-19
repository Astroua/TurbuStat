
'''
Test functions for bispec
'''

import numpy as np

from ..statistics import BiSpectrum, BiSpectrum_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testBispec():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_Bispec_method(self):
        self.tester = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).bispec1.bicoherence
        assert np.allclose(self.tester, self.computed_data['bispec_val'])

    def test_Bispec_distance(self):
    	self.tester = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric().distance
    	assert np.allclose(self.tester, self.computed_distances['bispec_distance'])