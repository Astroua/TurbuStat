# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt

from ..statistics.elliptical_powerlaw import (fit_elliptical_powerlaw,
                                              LogEllipticalPowerLaw2D,
                                              interval_transform,
                                              inverse_interval_transform)
from ..simulator import make_extended


def test_simple_ellipplaw():

    xvals = np.arange(1, 51)

    yvals = 5 * np.log10(xvals) + 2

    # Ellipticity is remapped to the real line. 1.0 is then inf.
    model = LogEllipticalPowerLaw2D(2, np.inf, 0.0, 5.)
    model_yvals = model(xvals, np.zeros_like(xvals))
    npt.assert_allclose(yvals, model_yvals)

    # Same answer for both y and x
    model_yvals = model(np.zeros_like(xvals), xvals)
    npt.assert_allclose(yvals, model_yvals)

    # Move away from circular. In the x-direction, transform as x / q
    # No rotation.
    # ellip set to 0.5, which maps to 0.0
    model_ellip = LogEllipticalPowerLaw2D(2, 0.0, 0.0, 5.)

    # This adds an offset, since the x-axis is squished by the ellipticity
    model_yvals = model_ellip(xvals, np.zeros_like(xvals))
    # Offset is q**index
    npt.assert_allclose(yvals + 5. * np.log10(0.5), model_yvals)

    # But the minor axis along y shouldn't have changed
    model_yvals = model_ellip(np.zeros_like(xvals), xvals)
    npt.assert_allclose(yvals, model_yvals)


@pytest.mark.parametrize(('plaw', 'ellip', 'theta'),
                         [(plaw, ellip, theta) for plaw in [2, 3, 4]
                          for ellip in [0.2, 0.5, 0.75, 0.9]
                          for theta in [np.pi / 4., np.pi / 2.,
                                        2 * np.pi / 3., np.pi]])
def test_simple_ellipplaw_2D_anisotropic(plaw, ellip, theta):

    # Must have ellip < 1 for this test. Just be sure...
    assert ellip < 1.

    imsize = 128

    # Generate a red noise model
    psd = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_fft=True)

    psd = np.abs(psd)**2

    # Initial guesses are based on the azimuthally-average spectrum, so it's
    # valid to give it good initial guesses for the index
    # Guess it is fairly elliptical. Tends not to be too sensitive to this.
    ellip_transf = interval_transform(ellip, 0, 1.)
    # We fit twice w/ thetas offset by pi / 2, so theta also should not be too
    # sensitive.
    p0 = (3.7, ellip_transf, np.pi / 2., plaw)

    yy, xx = np.mgrid[-imsize / 2:imsize / 2, -imsize / 2:imsize / 2]

    # Don't fit the 0, 0 point. It isn't defined by the model.
    valids = psd != 0.

    assert np.isfinite(psd[valids]).all()
    assert (psd[valids] > 0.).all()

    test_fit, test_stderr = \
        fit_elliptical_powerlaw(np.log10(psd[valids]),
                                xx[valids], yy[valids], p0,
                                bootstrap=False)[:2]

    # Do the parameters match?

    # Require the index to be within 0.1 of the actual,
    # the ellipticity to be within 0.02, and the theta to be within ~3 deg

    npt.assert_allclose(-plaw, test_fit[-1], atol=0.1)

    npt.assert_allclose(ellip, inverse_interval_transform(test_fit[1], 0, 1),
                        atol=0.01)

    # Theta can wrap by pi
    fit_theta = test_fit[2] % np.pi

    if np.abs(fit_theta - theta) > np.abs(fit_theta - theta + np.pi):
        theta = (theta - np.pi) % np.pi

    npt.assert_allclose(theta, fit_theta, atol=0.01)


@pytest.mark.parametrize('plaw', [2, 3, 4])
def test_simple_ellipplaw_2D_isotropic(plaw):

    # Must have ellip = 1 for this test. Just be sure...
    ellip = 1.

    # Theta doesn't matter, but we'll test for what happens when the
    # elliptical parameters are left free.
    theta = np.pi / 2.

    imsize = 256

    # Generate a red noise model
    psd = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_fft=True)

    psd = np.abs(psd)**2

    # Initial guesses are based on the azimuthally-average spectrum, so it's
    # valid to give it good initial guesses for the index
    # Guess it is fairly elliptical. Tends not to be too sensitive to this.
    ellip_transf = interval_transform(ellip, 0, 1.)
    # We fit twice w/ thetas offset by pi / 2, so theta also should not be too
    # sensitive.
    p0 = (3.7, ellip_transf, np.pi / 2., plaw)

    yy, xx = np.mgrid[-imsize / 2:imsize / 2, -imsize / 2:imsize / 2]

    # Don't fit the 0, 0 point. It isn't defined by the model.
    valids = psd != 0.

    assert np.isfinite(psd[valids]).all()
    assert (psd[valids] > 0.).all()

    # First fit with the elliptical parameters left free.
    test_fit, test_stderr = \
        fit_elliptical_powerlaw(np.log10(psd[valids]),
                                xx[valids], yy[valids], p0,
                                bootstrap=False)[:2]

    # Do the parameters match?

    # Require the index to be within 0.1 of the actual,
    # the ellipticity to be within 0.02, and the theta to be within ~3 deg

    npt.assert_allclose(-plaw, test_fit[-1], atol=0.1)

    npt.assert_allclose(ellip, inverse_interval_transform(test_fit[1], 0, 1),
                        atol=0.01)

    # Ensure fixing the elliptical parameters is working
    # And check that bootstrapping keeps those params fixed.
    test_fit, test_stderr = \
        fit_elliptical_powerlaw(np.log10(psd[valids]),
                                xx[valids], yy[valids], p0,
                                bootstrap=True, niters=2,
                                fix_ellip_params=True,
                                radial_weighting=True)[:2]

    # Should have a great constraint on the slope now
    npt.assert_allclose(-plaw, test_fit[-1], atol=0.001)

    # Transformed ellip should be inf
    assert np.isinf(test_fit[1])
    # Transforming back should give 1.0
    assert inverse_interval_transform(test_fit[1], 0, 1) == 1.

    # Theta should be the original
    assert theta == np.pi / 2.

    # Ellip param will be a NaN b/c it stays at inf
    assert np.isnan(test_stderr[1])

    # And theta should not move
    assert test_stderr[2] == 0.
