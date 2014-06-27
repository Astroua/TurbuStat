'''
Test functions for VCS
'''

import numpy as np

from ..statistics import VCS, VCS_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testVCS():
    #@classmethod
    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_VCS_method(self):
        self.tester = VCS(dataset1["cube"][0],dataset1["cube"][1])
        self.tester = self.tester.run()
        print self.tester.vel_freqs
        print self.computed_data['vcs_val']
        assert np.allclose(self.tester.vel_freqs, self.computed_data['vcs_val'])

    def test_VCS_distance(self):
        self.tester_dist = VCS_Distance(dataset1["cube"],dataset2["cube"])#, fiducial_model = self.tester)
        self.tester_dist = self.tester_dist.distance_metric().distance
        assert np.allclose(self.tester_dist, self.computed_distances['vcs_distance'])
