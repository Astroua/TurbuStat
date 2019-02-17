# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division


'''
Test functions for Cramer
'''

import numpy.testing as npt
import os

from ..statistics import Cramer_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_cramer():
    tester = \
        Cramer_Distance(dataset1["cube"],
                        dataset2["cube"],
                        noise_value1=0.1,
                        noise_value2=0.1).distance_metric(normalize=False,
                                                          verbose=True,
                                                          save_name='test.png')
    os.system("rm test.png")

    npt.assert_allclose(tester.data_matrix1,
                        computed_data["cramer_val"])
    npt.assert_almost_equal(tester.distance,
                            computed_distances['cramer_distance'])


def test_cramer_spatial_diff():

    small_data = dataset1["cube"][0][:, :26, :26]

    tester2 = Cramer_Distance(small_data, dataset2["cube"])
    tester2.distance_metric(normalize=False)
    tester3 = Cramer_Distance(dataset2["cube"], small_data)
    tester3.distance_metric(normalize=False)

    npt.assert_almost_equal(tester2.distance, tester3.distance)
