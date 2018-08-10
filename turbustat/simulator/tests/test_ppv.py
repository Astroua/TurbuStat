
from ..make_cube import make_ppv
from ..gen_field import make_3dfield

import numpy as np
import numpy.testing as npt
import astropy.units as u


def test_ppv():
    '''
    Ensure the column density matches the expected value in the output cube
    '''

    # Need a large enough field to have good statistics
    velocity = make_3dfield(128, powerlaw=3.5, amp=5.e3) * u.m / u.s

    density = np.ones_like(velocity.value) * u.cm**-3

    cube, spec_axis = make_ppv(velocity[:, :2, :2], density[:, :2, :2],
                               vel_disp=np.std(velocity), T=100 * u.K,
                               return_hdu=False,
                               chan_width=500 * u.m / u.s)

    NHI_exp = (1 * u.cm**-3) * (1 * u.pc).to(u.cm)

    chan_width = np.diff(spec_axis[:2])[0].to(u.km / u.s)

    # Moment 0 in K km/s
    mom0 = cube.sum(0) * chan_width

    # Convert to cm^-2
    NHI_cube = mom0 * 1.823e18 * u.cm**-2 / (u.K * u.km / u.s)

    assert NHI_exp.unit == NHI_cube.unit

    npt.assert_allclose(NHI_exp.value, NHI_cube.value, rtol=1e-4)
