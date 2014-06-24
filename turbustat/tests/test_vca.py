'''
Test functions for VCA
'''

import numpy as np

from ..statistics import VCA, VCA_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testVCA():

	def __init__(self):
		self.dataset1 = dataset1
		self.dataset2 = dataset2
		self.computed_data = computed_data
		self.computed_distances = computed_distances
		self.tester = None

	def test_VCA_method(self):
		self.tester = VCA(dataset1["cube"][0],dataset1["cube"][1])
		self.tester.run()
		print self.tester.ps1D
		print self.computed_distances['vca_val']
		assert np.allclose(self.tester.ps1D, self.computed_data['vca_val'])

	def test_VCA_distance(self):
		self.tester_dist = VCA_Distance(dataset1["cube"],dataset2["cube"], fiducial_model = self.tester).distance_metric().distance

		assert np.allclose(self.tester_dist, self.computed_distances['vca_distance'])