# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
from scipy.ndimage import zoom
import astropy.units as u
import os
from astropy.io import fits

from ..statistics import SCF, SCF_Distance
from ._testing_data import (dataset1, dataset2, computed_data,
                            computed_distances, make_extended, assert_between)


def test_SCF_method():
    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='continuous')

    assert np.allclose(tester.scf_surface, computed_data['scf_val'])
    npt.assert_array_almost_equal(tester.scf_spectrum,
                                  computed_data["scf_spectrum"])
    npt.assert_almost_equal(tester.slope, computed_data["scf_slope"])
    npt.assert_almost_equal(tester.slope2D, computed_data["scf_slope2D"])

    # Test the save and load
    tester.save_results(keep_data=False)
    tester.load_results("scf_output.pkl")

    # Remove the file
    os.remove("scf_output.pkl")

    assert np.allclose(tester.scf_surface, computed_data['scf_val'])
    npt.assert_array_almost_equal(tester.scf_spectrum,
                                  computed_data["scf_spectrum"])
    npt.assert_almost_equal(tester.slope, computed_data["scf_slope"])


def test_SCF_method_fitlimits():
    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='continuous', xlow=1.5 * u.pix,
               xhigh=4.5 * u.pix)

    npt.assert_almost_equal(tester.slope, computed_data["scf_slope_wlimits"])
    npt.assert_almost_equal(tester.slope2D,
                            computed_data["scf_slope_wlimits_2D"])


def test_SCF_method_fitlimits_units():

    distance = 250 * u.pc

    xlow = 1.5 * u.pix
    xhigh = 4.5 * u.pix

    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='continuous', xlow=xlow,
               xhigh=xhigh, fit_2D=False)

    npt.assert_almost_equal(tester.slope, computed_data["scf_slope_wlimits"])

    xlow = xlow.value * dataset1['cube'][1]['CDELT2'] * u.deg
    xhigh = xhigh.value * dataset1['cube'][1]['CDELT2'] * u.deg

    tester2 = SCF(dataset1["cube"], size=11)
    tester2.run(boundary='continuous', xlow=xlow,
                xhigh=xhigh, fit_2D=False)

    npt.assert_almost_equal(tester2.slope, computed_data["scf_slope_wlimits"])

    xlow = xlow.to(u.rad).value * distance
    xhigh = xhigh.to(u.rad).value * distance

    tester3 = SCF(dataset1["cube"], size=11, distance=distance)
    tester3.run(boundary='continuous', xlow=xlow,
                xhigh=xhigh, fit_2D=False)

    npt.assert_almost_equal(tester3.slope, computed_data["scf_slope_wlimits"])


def test_SCF_method_noncont_boundary():
    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='cut')

    assert np.allclose(tester.scf_surface,
                       computed_data['scf_val_noncon_bound'])


def test_SCF_noninteger_shift():
    # Not testing against anything, just make sure it runs w/o issue.
    rolls = np.array([-4.5, -3.0, -1.5, 0, 1.5, 3.0, 4.5]) * u.pix
    tester_nonint = \
        SCF(dataset1["cube"], roll_lags=rolls)
    tester_nonint.run()


def test_SCF_nonpixelunit_shift():
    # Not testing against anything, just make sure it runs w/o issue.
    rolls = np.array([-4.5, -3.0, -1.5, 0, 1.5, 3.0, 4.5]) * u.pix

    # Convert the rolls to angular units
    ang_rolls = rolls.value * np.abs(dataset1['cube'][1]["CDELT2"]) * u.deg

    tester_angroll = \
        SCF(dataset1["cube"], roll_lags=ang_rolls)
    tester_angroll.run()

    # And one passing physical distances
    dist = 250 * u.pc
    phys_rolls = ang_rolls.to(u.rad).value * dist
    tester_physroll = \
        SCF(dataset1["cube"], roll_lags=phys_rolls, distance=dist)
    tester_physroll.run()

    # The SCF surfaces should be the same.
    assert np.allclose(tester_angroll.scf_surface,
                       tester_physroll.scf_surface)


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


def test_SCF_azimlimits():
    '''
    The slopes with azimuthal constraints should be the same. When elliptical,
    the power will be different along the different directions, but the slope
    should remain the same.
    '''

    test = SCF(dataset1["cube"], size=11)
    test.run(boundary='continuous',
             radialavg_kwargs={"theta_0": 0 * u.deg,
                               "delta_theta": 40 * u.deg})

    test2 = SCF(dataset1["cube"], size=11)
    test2.run(boundary='continuous',
              radialavg_kwargs={"theta_0": 90 * u.deg,
                                "delta_theta": 40 * u.deg})

    test3 = SCF(dataset1["cube"], size=11)
    test3.run(boundary='continuous',
              radialavg_kwargs={})

    # Ensure slopes are consistent to within 5%
    npt.assert_allclose(test3.slope, test.slope, atol=5e-3)
    npt.assert_allclose(test3.slope, test2.slope, atol=5e-3)


@pytest.mark.parametrize('theta',
                         [0., np.pi / 4., np.pi / 2., 7 * np.pi / 8.])
def test_scf_fit2D(theta):
    '''
    Since test_elliplaw tests everything, only check for consistent theta
    here.
    '''

    imsize = 512
    ellip = 0.5
    plaw = 4.

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_psd=False)[np.newaxis, ...]

    test = SCF(fits.PrimaryHDU(img), size=21)
    test.run(fit_2D=True)

    try:
        npt.assert_allclose(theta, test.theta2D,
                            atol=0.12)
    except AssertionError:
        npt.assert_allclose(theta, test.theta2D - np.pi,
                            atol=0.12)

