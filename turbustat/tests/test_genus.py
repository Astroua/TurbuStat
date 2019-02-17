# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

'''
Test functions for Genus
'''

import numpy as np
import numpy.testing as npt
import astropy.units as u
from copy import copy
import os

from ..statistics import Genus_Distance, Genus
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_Genus_method():

    smooth_scales = np.linspace(1.0, 0.1 * min(dataset1["moment0"][0].shape), 5)

    tester = Genus(dataset1["moment0"],
                   smoothing_radii=smooth_scales)
    tester.run(match_kernel=True, verbose=True, save_name='test.png')

    os.system("rm test.png")

    assert np.allclose(tester.genus_stats,
                       computed_data['genus_val'])

    # Test loading and saving
    tester.save_results("genus_output.pkl", keep_data=False)

    saved_tester = Genus.load_results("genus_output.pkl")

    # Remove the file
    os.remove("genus_output.pkl")

    assert np.allclose(saved_tester.genus_stats,
                       computed_data['genus_val'])


def test_Genus_method_headerbeam():

    mom0 = copy(dataset1["moment0"])
    mom0[1]["BMAJ"] = 1.0

    # Just ensuring these run without issue.
    smooth_scales = np.linspace(1.0, 0.1 * min(mom0[0].shape), 5)

    tester = Genus(mom0, smoothing_radii=smooth_scales)
    tester.run(use_beam=True, match_kernel=True)

    tester2 = Genus(mom0, smoothing_radii=smooth_scales)
    tester2.run(use_beam=True, min_size=1.0 * u.deg**2,
                match_kernel=True)

    npt.assert_allclose(tester.genus_stats, tester2.genus_stats)


def test_Genus_method_value_vs_perc():

    min_perc1 = 20
    min_val1 = np.percentile(dataset1['moment0'][0], min_perc1)

    max_perc1 = 90
    max_val1 = np.percentile(dataset1['moment0'][0], max_perc1)

    smooth_scales = np.linspace(1.0, 0.1 * min(dataset1["moment0"][0].shape),
                                5)

    tester = Genus(dataset1['moment0'], lowdens_percent=min_perc1,
                   highdens_percent=max_perc1,
                   smoothing_radii=smooth_scales)
    tester.run(match_kernel=True)

    tester2 = Genus(dataset1['moment0'], min_value=min_val1,
                    max_value=max_val1,
                    smoothing_radii=smooth_scales)
    tester2.run(match_kernel=True)

    npt.assert_allclose(tester.genus_stats, tester2.genus_stats)


def test_Genus_method_smoothunits():

    distance = 250 * u.pc

    radii = np.linspace(1.0, 0.1 * min(dataset1['moment0'][0].shape), 5) * u.pix
    tester = Genus(dataset1["moment0"], smoothing_radii=radii)
    tester.run(match_kernel=True)

    radii = radii.value * dataset1['moment0'][1]['CDELT2'] * u.deg
    tester2 = Genus(dataset1["moment0"], smoothing_radii=radii)
    tester2.run(match_kernel=True)

    radii = radii.to(u.rad).value * distance
    tester3 = Genus(dataset1["moment0"], smoothing_radii=radii,
                    distance=distance)
    tester3.run(match_kernel=True)

    npt.assert_allclose(tester.genus_stats, tester2.genus_stats)
    npt.assert_allclose(tester.genus_stats, tester3.genus_stats)


def test_Genus_distance():
    tester_dist = \
        Genus_Distance(dataset1["moment0"],
                       dataset2["moment0"],
                       lowdens_percent=20,
                       genus_kwargs=dict(match_kernel=True))
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['genus_distance'])
