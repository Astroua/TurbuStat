# Licensed under an MIT open source license - see LICENSE


'''
Test functions for SCF
'''

import pytest

import numpy as np
import numpy.testing as npt
from scipy.ndimage import zoom

from ..statistics import SCF, SCF_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_SCF_method():
    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='continuous')

    assert np.allclose(tester.scf_surface, computed_data['scf_val'])
    npt.assert_array_almost_equal(tester.scf_spectrum,
                                  computed_data["scf_spectrum"])
    npt.assert_almost_equal(tester.slope, computed_data["scf_slope"])


def test_SCF_method_noncont_boundary():
    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='cut')

    assert np.allclose(tester.scf_surface,
                       computed_data['scf_val_noncon_bound'])


def test_SCF_noninteger_shift():
    # Not testing against anything, just make sure it runs w/o issue.
    rolls = np.array([-4.5, -3.0, -1.5, 0, 1.5, 3.0, 4.5])
    tester_nonint = \
        SCF(dataset1["cube"], roll_lags=rolls)
    tester_nonint.run()


def test_SCF_distance():
    tester_dist = \
        SCF_Distance(dataset1["cube"],
                     dataset2["cube"], size=11).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['scf_distance'])


def test_SCF_regrid_distance():
    hdr = dataset1["cube"][1].copy()
    hdr["CDELT2"] = 0.5 * hdr["CDELT2"]
    hdr["CDELT1"] = 0.5 * hdr["CDELT1"]
    cube = zoom(dataset1["cube"][0], (1, 2, 2))

    tester_dist_zoom = \
        SCF_Distance([cube, hdr], dataset1["cube"],
                     size=11).distance_metric()

    # Based on the fiducial values, the distance should be
    # at least less than this.
    fid_dist = 0.02

    assert tester_dist_zoom.distance < fid_dist
