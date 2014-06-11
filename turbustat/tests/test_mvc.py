
'''
Test functions for MVC
'''

import numpy as np

from ..statistics import MVC, MVC_distance
from ._testing_data import dataset1, dataset2

class testMVC():

    def setup_class():
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.computed_data = None

    def test_MVC_method():
        tester = MVC(dataset1["centroid"][0], dataset1["moment0"][0], dataset1["linewidth"][0],
                     dataset1["centroid"][1])

        tester.run()

        np.testing.assert_array_equal(tester.ps1D, self.computed_data.ps1D)
