
'''
Test functions for Kurtosis
'''

import numpy as np

from ..statistics import  StatMoments, StatMomentsDistance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testKurtSkew():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_Kurtosis_method(self):
        self.tester = StatMoments(dataset1["integrated_intensity"][0], 5)
        self.tester.run()
        assert np.allclose(self.tester.kurtosis_array, self.computed_data['kurtosis_val'])

    def test_Kurtosis_distance(self):
        self.tester_dist = StatMomentsDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0], 5).distance_metric(verbose=False)
        assert np.allclose(self.tester_dist.kurtosis_distance, self.computed_distances['kurtosis_distance'])

    def test_Skewness_method(self):
        assert np.allclose(self.tester.skewness_array , self.computed_data['skewness_val'])

    def test_Skewness_distance(self):
        assert np.allclose(self.tester_dist.distance_metric().skewness_distance, self.computed_distances['skewness_distance'])
