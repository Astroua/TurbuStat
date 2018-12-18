
from ..make_cube import make_ppv
from ..gen_field import make_3dfield

import pytest
import numpy as np
import numpy.testing as npt
import astropy.units as u
import astropy.constants as c

try:
    from spectral_cube import SpectralCube
    SPECCUBE_INSTALL = True
except ImportError:
    SPECCUBE_INSTALL = False


@pytest.mark.skipif("not SPECCUBE_INSTALL")
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_ppv(axis):
    '''
    Ensure the column density matches the expected value in the output cube
    '''

    # Number of samples to take along each non-projected dimension
    size = 16

    twod_slice = [slice(0, size), slice(0, size)]
    threed_slice = [slice(0, size), slice(0, size)]
    threed_slice.insert(axis, slice(None))

    # Need a large enough field to have good statistics
    velocity = make_3dfield(128, powerlaw=3.5, amp=5.e3) * u.m / u.s

    density = np.ones_like(velocity.value) * u.cm**-3

    cube_hdu = make_ppv(velocity[threed_slice], density[threed_slice],
                        los_axis=axis,
                        vel_disp=np.std(velocity, axis=axis)[twod_slice].max(),
                        T=100 * u.K,
                        return_hdu=True)
                        # chan_width=500 * u.m / u.s)

    cube = SpectralCube.read(cube_hdu)

    chan_width = np.abs(np.diff(cube.spectral_axis)[0]).to(u.m / u.s)

    NHI_exp = (1 * u.cm**-3) * (1 * u.pc).to(u.cm)

    # Moment 0 in K km/s
    mom0 = cube.moment0().to(u.K * u.km / u.s)

    # Convert to cm^-2
    NHI_cube = mom0 * 1.823e18 * u.cm**-2 / (u.K * u.km / u.s)

    assert NHI_exp.unit == NHI_cube.unit

    # Expected is 3.0854e18. Check if it is within 1e16
    npt.assert_allclose(NHI_exp.value, NHI_cube.value, rtol=1e-3)

    v_therm = np.sqrt(c.k_B * 100 * u.K / (1.4 * c.m_p)).to(u.km / u.s)

    # Compare centroids

    raw_centroid = ((velocity[threed_slice] * density[threed_slice]).sum(axis) /
                    (density[threed_slice]).sum(axis)).to(u.km / u.s)
    mom1 = cube.moment1().to(u.km / u.s)

    npt.assert_allclose(raw_centroid.value, mom1.value,
                        atol=chan_width.value / 2.)

    # Rough comparison of line width to the velocity field std.
    # Very few samples, so this is only a rough check

    # Correct the measured line widths for thermal broadening and broadening
    # from finite channel widths
    lwidth = np.sqrt(cube.linewidth_sigma().to(u.km / u.s)**2 -
                     chan_width**2 - v_therm**2)
    vel_std = np.std(velocity, axis=axis)[twod_slice].to(u.km / u.s)

    npt.assert_allclose(vel_std.value, lwidth.value, atol=0.2)


@pytest.mark.skipif("not SPECCUBE_INSTALL")
@pytest.mark.xfail(raises=ValueError)
def test_ppv_negative_density():
    '''
    Negative densities should give a ValueError
    '''

    # Number of samples to take along each non-projected dimension
    size = 16

    axis = 0

    twod_slice = [slice(0, size), slice(0, size)]
    threed_slice = [slice(0, size), slice(0, size)]
    threed_slice.insert(axis, slice(None))

    # Need a large enough field to have good statistics
    velocity = make_3dfield(128, powerlaw=3.5, amp=5.e3) * u.m / u.s

    density = np.ones_like(velocity.value) * u.cm**-3

    density[4, 4, 4] *= -1.

    cube_hdu = make_ppv(velocity[threed_slice], density[threed_slice],
                        los_axis=axis,
                        vel_disp=np.std(velocity, axis=axis)[twod_slice].max(),
                        T=100 * u.K,
                        return_hdu=True)
