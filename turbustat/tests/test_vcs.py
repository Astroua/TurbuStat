'''
Test functions for VCS
'''

import numpy as np

from ..statistics import VCS, VCS_Distance
from ._testing_data import dataset1, dataset2, computed_data, computed_distances

class testVCS():

    def __init__(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = computed_data
        self.computed_distances = computed_distances
        self.tester = None

    def test_VCS_method(self):
        self.tester = VCS_Distance(dataset1["cube"],dataset2["cube"]).vcs1.ps1D
        assert np.allclose(self.tester, self.computed_data['vcs_val'])

    def test_VCS_distance(self):
        self.tester = VCS_Distance(dataset1["cube"],dataset2["cube"]).distance_metric().distance
        assert np.allclose(self.tester, self.computed_distances['vcs_distance'])