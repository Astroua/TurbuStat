# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division


'''
Tests for Dendrogram statistics
'''

import numpy as np
import numpy.testing as npt
import os

from ..statistics import Dendrogram_Stats, DendroDistance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


min_deltas = np.logspace(-1.5, 0.5, 40)


def test_DendroStat():

    tester = Dendrogram_Stats(dataset1["cube"],
                              min_deltas=min_deltas)
    tester.run(periodic_bounds=False, verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_allclose(tester.numfeatures,
                        computed_data["dendrogram_val"])

    # Test loading and saving
    tester.save_results("dendrogram_stats_output.pkl", keep_data=False)

    saved_tester = Dendrogram_Stats.load_results("dendrogram_stats_output.pkl")

    # Remove the file
    os.remove("dendrogram_stats_output.pkl")

    npt.assert_allclose(saved_tester.numfeatures,
                        computed_data["dendrogram_val"])


def test_DendroStat_periodic():

    tester = Dendrogram_Stats(dataset1["cube"],
                              min_deltas=min_deltas)
    tester.run(periodic_bounds=True)

    npt.assert_allclose(tester.numfeatures,
                        computed_data["dendrogram_periodic_val"])


def test_DendroDistance():

    tester_dist = \
        DendroDistance(dataset1["cube"],
                       dataset2["cube"],
                       min_deltas=min_deltas,
                       dendro_kwargs=dict(periodic_bounds=False))
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test*.png")

    npt.assert_almost_equal(tester_dist.histogram_distance,
                            computed_distances["dendrohist_distance"])
    npt.assert_almost_equal(tester_dist.num_distance,
                            computed_distances["dendronum_distance"])

    # With Dendrogram_Stats as inputs

    tester_dist2 = \
        DendroDistance(tester_dist.dendro1,
                       tester_dist.dendro2,
                       min_deltas=min_deltas,
                       dendro_kwargs=dict(periodic_bounds=False))
    tester_dist2.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist2.histogram_distance,
                            computed_distances["dendrohist_distance"])
    npt.assert_almost_equal(tester_dist2.num_distance,
                            computed_distances["dendronum_distance"])

    # With uncomputed Dendrogram_Stats
    tester = Dendrogram_Stats(dataset1["cube"],
                              min_deltas=min_deltas)
    tester2 = Dendrogram_Stats(dataset2["cube"],
                               min_deltas=min_deltas)

    tester_dist3 = \
        DendroDistance(tester,
                       tester2,
                       min_deltas=min_deltas,
                       dendro_kwargs=dict(periodic_bounds=False))
    tester_dist3.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist3.histogram_distance,
                            computed_distances["dendrohist_distance"])
    npt.assert_almost_equal(tester_dist3.num_distance,
                            computed_distances["dendronum_distance"])