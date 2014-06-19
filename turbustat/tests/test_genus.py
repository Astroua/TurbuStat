'''
Test functions for Genus
'''

import numpy as np

from ..statistics import Genus, GenusDistance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testGenus():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_Genus_method(self):
        self.tester = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).genus1.thresholds
        assert np.allclose(self.tester, self.computed_data['genus_val'])

    def test_Genus_distance(self):
        self.tester = GenusDistance(dataset1["integrated_intensity"][0], dataset2["integrated_intensity"][0]).distance_metric().distance
        assert np.allclose(self.tester, self.computed_distances['genus_distance'])