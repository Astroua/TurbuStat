# Licensed under an MIT open source license - see LICENSE

'''
Test functions for PCA
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import PCA, PCA_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testPCA(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_PCA_method(self):
        self.tester = PCA(dataset1["cube"][0], n_eigs=50)
        self.tester.run(normalize=True)
        assert np.allclose(self.tester.eigvals, computed_data['pca_val'])

    def test_PCA_distance(self):
        self.tester_dist = \
            PCA_Distance(dataset1["cube"][0],
                         dataset2["cube"][0], normalize=True).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['pca_distance'])
