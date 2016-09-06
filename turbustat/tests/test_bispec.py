# Licensed under an MIT open source license - see LICENSE


'''
Test functions for bispec
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import BiSpectrum, BiSpectrum_Distance
from ._testing_data import dataset1,\
    dataset2, computed_data, computed_distances


class testBispec(TestCase):

    def setupUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_Bispec_method(self):
        self.tester = BiSpectrum(dataset1["moment0"])
        self.tester.run()
        assert np.allclose(self.tester.bicoherence,
                           computed_data['bispec_val'])

    def test_Bispec_distance(self):
        self.tester_dist = \
            BiSpectrum_Distance(dataset1["moment0"],
                                dataset2["moment0"])
        self.tester_dist.distance_metric()

        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['bispec_distance'])
