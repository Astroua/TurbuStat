
'''
Test function for Wavelet
'''

import numpy as np
from ..statistics import wavelets, Wavelet_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testWavelet():

	def __init__(self):
		self.dataset1 = dataset1
		self.dataset2 = dataset2
		self.computed_data = computed_data
		self.computed_distances = computed_distances
		self.tester = None

	def test_Wavelet_method(self):
		self.tester = wt2D(dataset1["integrated_intensity"][0], np.logspace(np.log10(round((5. / 3.), 3)), np.log10(min(dataset1["integrated_intensity"][0].shape)), 50))
		assert np.allclose(self.tester.Wf, self.computed_data['wavelet_val'])

	def test_Wavelet_distance(self):
		self.tester_dist = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"], fiducial_model = self.tester).distance_metric().distance
		assert np.allclose(self.tester_dist, self.computed_distances['wavelet_distance'])
