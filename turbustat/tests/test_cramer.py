
'''
Test functions for Cramer
'''

import numpy as np

from ..statistics import cramer, Cramer_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testCramer():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_Cramer_distance(self):
    	self.tester = Cramer_Distance(dataset1["cube"][0], dataset2["cube"][0])
        self.tester = self.tester.distance_metric()
    	assert self.tester.distance == self.computed_distances['cramer_distance']
