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
                            computed_distances)
from ..simulator import make_extended


def test_SCF_method():
    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='continuous', verbose=True, save_name='test.png')
    os.system("rm test.png")

    # Test fitting with bootstrapping
    tester.fit_plaw(bootstrap=True)

    assert np.allclose(tester.scf_surface, computed_data['scf_val'])
    npt.assert_array_almost_equal(tester.scf_spectrum,
                                  computed_data["scf_spectrum"])
    npt.assert_almost_equal(tester.slope, computed_data["scf_slope"],
                            decimal=3)
    npt.assert_almost_equal(tester.slope2D, computed_data["scf_slope2D"],
                            decimal=3)

    # Test the save and load
    tester.save_results("scf_output.pkl", keep_data=False)
    saved_tester = SCF.load_results("scf_output.pkl")

    # Remove the file
    os.remove("scf_output.pkl")

    assert np.allclose(saved_tester.scf_surface, computed_data['scf_val'])
    npt.assert_array_almost_equal(saved_tester.scf_spectrum,
                                  computed_data["scf_spectrum"])
    npt.assert_almost_equal(saved_tester.slope, computed_data["scf_slope"],
                            decimal=3)


def test_SCF_method_fitlimits():
    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='continuous', xlow=1.5 * u.pix,
               xhigh=4.5 * u.pix)

    npt.assert_almost_equal(tester.slope, computed_data["scf_slope_wlimits"],
                            decimal=3)
    npt.assert_almost_equal(tester.slope2D,
                            computed_data["scf_slope_wlimits_2D"], decimal=3)


def test_SCF_method_fitlimits_units():

    distance = 250 * u.pc

    xlow = 1.5 * u.pix
    xhigh = 4.5 * u.pix

    tester = SCF(dataset1["cube"], size=11)
    tester.run(boundary='continuous', xlow=xlow,
               xhigh=xhigh, fit_2D=False)

    npt.assert_almost_equal(tester.slope, computed_data["scf_slope_wlimits"],
                            decimal=3)

    xlow = xlow.value * dataset1['cube'][1]['CDELT2'] * u.deg
    xhigh = xhigh.value * dataset1['cube'][1]['CDELT2'] * u.deg

    tester2 = SCF(dataset1["cube"], size=11)
    tester2.run(boundary='continuous', xlow=xlow,
                xhigh=xhigh, fit_2D=False)

    npt.assert_almost_equal(tester2.slope, computed_data["scf_slope_wlimits"],
                            decimal=3)

    xlow = xlow.to(u.rad).value * distance
    xhigh = xhigh.to(u.rad).value * distance

    tester3 = SCF(dataset1["cube"], size=11, distance=distance)
    tester3.run(boundary='continuous', xlow=xlow,
                xhigh=xhigh, fit_2D=False)

    npt.assert_almost_equal(tester3.slope, computed_data["scf_slope_wlimits"],
                            decimal=3)


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
                     dataset2["cube"], size=11)
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['scf_distance'])

    # With pre-computed SCF classes

    tester_dist2 = \
        SCF_Distance(tester_dist.scf1,
                     tester_dist.scf2, size=11)
    tester_dist2.distance_metric()

    npt.assert_almost_equal(tester_dist2.distance,
                            computed_distances['scf_distance'])

    # With fresh SCF instances
    tester = SCF(dataset1["cube"], size=11)
    tester2 = SCF(dataset2["cube"], size=11)

    tester_dist3 = \
        SCF_Distance(tester,
                     tester2, size=11)
    tester_dist3.distance_metric()

    npt.assert_almost_equal(tester_dist3.distance,
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


@pytest.mark.parametrize(('theta', 'ellip'),
                         [(theta, ellip) for theta in
                         [0., np.pi / 4., 7 * np.pi / 8.]
                          for ellip in [0.2, 0.8]])
def test_scf_fit2D(theta, ellip):
    '''
    Since test_elliplaw tests everything, only check for consistent theta
    here.
    '''

    nchans = 10
    imsize = 128
    plaw = 4.

    # Generate a red noise model
    cube = np.empty((nchans, imsize, imsize))

    for i in range(nchans):
        cube[i] = make_extended(imsize, powerlaw=plaw, ellip=ellip,
                                theta=theta)

    test = SCF(fits.PrimaryHDU(cube), size=11)
    test.run(fit_2D=True, xhigh=7 * u.pix)

    # TODO: Need to return to this. The fit can vary by +/-0.2, but the theta
    # parameter is correct. The SCF surface is only a single power-law over a
    # limited range, despite the data being generated by a single 2D power-law

    # This also appears to matter which version of astropy fitting is being
    # used... The limited size of the 2D plane is at least partially the
    # cause of the uncertainty.

    try:
        npt.assert_allclose(theta, test.theta2D,
                            atol=0.2)
    except AssertionError:
        npt.assert_allclose(theta, test.theta2D - np.pi,
                            atol=0.2)
