
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
		self.tester = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).wt1.Wf
		assert np.allclose(self.tester, self.computed_data['wavelet_val'])

	def test_Wavelet_distance(self):
		self.tester = Wavelet_Distance(dataset1["integrated_intensity"], dataset2["integrated_intensity"]).distance_metric().distance
		assert np.allclose(self.tester, self.computed_distances['wavelet_distance'])