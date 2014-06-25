
'''
Test functions for MVC
'''

import numpy as np

from ..statistics import MVC, MVC_distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testMVC():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_MVC_method(self):
        self.tester = MVC(dataset1["centroid"][0], dataset1["moment0"][0], dataset1["linewidth"][0], dataset1["centroid"][1])
        self.tester.run()
        print(self.tester.ps1D)
        print(self.computed_data['mvc_val'])
        assert np.allclose(self.tester.ps1D, self.computed_data['mvc_val'])

    def test_MVC_distance(self):
        self.tester_dist = MVC_distance(dataset1, dataset2, fiducial_model = self.tester).distance_metric().distance
        assert np.allclose(self.tester_dist, self.computed_distances['mvc_distance'])

