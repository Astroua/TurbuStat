'''
Test functions for PCA
'''

import numpy as np

from ..statistics import PCA, PCA_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testPCA():
    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_PCA_method(self):
        self.tester = PCA(dataset1["cube"][0], n_eigs = 50)
        self.tester.run()
        assert np.allclose(self.tester.eigvals, self.computed_data['pca_val'])

    def test_PCA_distance(self):
        self.tester_dist = PCA_Distance(dataset1["cube"][0],dataset2["cube"][0],fiducial_model=self.tester).distance_metric().distance
        assert np.allclose(self.tester_dist, self.computed_distances['pca_distance'])
