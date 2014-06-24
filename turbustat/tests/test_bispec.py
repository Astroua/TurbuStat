
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
        self.tester = BiSpectrum(dataset1["integrated_intensity"][0], dataset1["integrated_intensity"][1])
        self.tester.run()
        assert np.allclose(self.tester.bicoherence, self.computed_data['bispec_val'])

    def test_Bispec_distance(self):
        self.tester_dist = BiSpectrum_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"], fiducial_model = self.tester).distance_metric().distance
    	assert np.allclose(self.tester_dist, self.computed_distances['bispec_distance'])
