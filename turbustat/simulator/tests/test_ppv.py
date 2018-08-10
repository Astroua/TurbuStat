
from ..make_cube import make_ppv
from ..gen_field import make_3dfield

import pytest
import numpy as np
import numpy.testing as npt
import astropy.units as u

try:
    from spectral_cube import SpectralCube
    SPECCUBE_INSTALL = True
except ImportError:
    SPECCUBE_INSTALL = False


@pytest.mark.skipif("not SPECCUBE_INSTALL")
def test_ppv():
    '''
    Ensure the column density matches the expected value in the output cube
    '''

    # Need a large enough field to have good statistics
    velocity = make_3dfield(128, powerlaw=3.5, amp=5.e3) * u.m / u.s

    density = np.ones_like(velocity.value) * u.cm**-3

    cube_hdu = make_ppv(velocity[:, :2, :2], density[:, :2, :2],
                        vel_disp=np.std(velocity), T=100 * u.K,
                        return_hdu=True,
                        chan_width=500 * u.m / u.s)

    cube = SpectralCube.read(cube_hdu)

    NHI_exp = (1 * u.cm**-3) * (1 * u.pc).to(u.cm)

    # Moment 0 in K km/s
    mom0 = cube.moment0().to(u.K * u.km / u.s)

    # Convert to cm^-2
    NHI_cube = mom0 * 1.823e18 * u.cm**-2 / (u.K * u.km / u.s)

    assert NHI_exp.unit == NHI_cube.unit

    npt.assert_allclose(NHI_exp.value, NHI_cube.value, rtol=1e-4)

    # Rough comparison of line width to the velocity field std.
    # Very few samples, so this is only a rough check
    lwidth = cube.linewidth_sigma().to(u.km / u.s)
    vel_std = np.std(velocity).to(u.km / u.s)

    npt.assert_allclose(vel_std.value, lwidth.value, rtol=0.2)

    # Compare centroids

    raw_centroid = ((velocity[:, :2, :2] * density[:, :2, :2]).sum(0) /
                    (density[:, :2, :2]).sum(0)).to(u.km / u.s)
    mom1 = cube.moment1().to(u.km / u.s)

    npt.assert_allclose(raw_centroid.value, mom1.value, rtol=1e-3)
