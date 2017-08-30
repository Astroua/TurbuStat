# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy.testing as npt
import astropy.units as u

from ..statistics import MVC, MVC_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_MVC_method():
    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester.run()

    npt.assert_allclose(tester.ps1D, computed_data['mvc_val'])
    npt.assert_almost_equal(tester.slope, computed_data['mvc_slope'])
    npt.assert_almost_equal(tester.slope2D, computed_data['mvc_slope2D'])


def test_MVC_method_fitlimits():

    distance = 250 * u.pc

    low_cut = 0.02 / u.pix
    high_cut = 0.1 / u.pix

    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.value / (dataset1['cube'][1]['CDELT2'] * u.deg)
    high_cut = high_cut.value / (dataset1['cube'][1]['CDELT2'] * u.deg)

    tester2 = MVC(dataset1["centroid"],
                  dataset1["moment0"],
                  dataset1["linewidth"],
                  dataset1["centroid"][1])
    tester2.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.to(u.rad**-1).value / distance
    high_cut = high_cut.to(u.rad**-1).value / distance

    tester3 = MVC(dataset1["centroid"],
                  dataset1["moment0"],
                  dataset1["linewidth"],
                  dataset1["centroid"][1], distance=distance)
    tester3.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_MVC_distance():
    tester_dist = \
        MVC_Distance(dataset1, dataset2).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['mvc_distance'])
