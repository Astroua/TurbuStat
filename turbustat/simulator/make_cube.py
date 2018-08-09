
import numpy as np
import astropy.constants as co
import astropy.units as u
from itertools import product
from warnings import warn
from multiprocessing import Pool
from astropy.utils.console import ProgressBar
from astropy.io import fits

from ..io.sim_tools import create_cube_header


SQRT_2PI = np.sqrt(2 * np.pi)
conv_to_K = 1.823e13 * u.cm**-2 / (u.K * u.cm / u.s)


def make_ppv(vel_field, dens_field, los_axis=0,
             m=1.4 * co.m_p, T=4000 * u.K, los_length=1 * u.pc,
             vel_disp=None, chan_width=None, v_min=None, v_max=None,
             threads=1, max_chan=1000,
             vel_struct_index=0.5, verbose=False,
             return_hdu=True, pixel_ang_scale=1 * u.arcmin,
             restfreq=1.42 * u.GHz):
    '''
    Generate a mock, optically-thin PPV cube from a given velocity and
    density field.

    Parameters
    ----------
    los_length : `~astropy.units.Unit` length, optional
        Set the total physical length of the density and velocity fields
        along the line-of-sight. Defaults to 1 pc.
    '''

    v_therm_sq = (co.k_B * T / m).to(vel_field.unit**2)
    v_therm = np.sqrt(v_therm_sq)

    # Setup velocity axis
    if v_min is None:
        v_min = vel_field.min() - 4 * v_therm
    else:
        if not v_min.unit.is_equivalent(u.km / u.s):

            raise u.UnitsError("v_min must be given in velocity units.")
    if v_max is None:
        v_max = vel_field.max() + 4 * v_therm
    else:
        if not v_max.unit.is_equivalent(u.km / u.s):
            raise u.UnitsError("v_max must be given in velocity units.")

    if v_min > v_max:
        raise ValueError("v_min > v_max is not allowed.")

    del_v = v_max - v_min

    # m=1 assumption.
    if vel_struct_index is None:
        warn("Setting channels for velocity structure index of 0.5.")
        vel_struct_index = 0.5

    if chan_width is None:
        # Eq. 13 from Esquivel+2003. Number of channels to recover the thin
        # slice regime down to 2 pix.

        # Take the effective width to be ~1/5 of the effective channel width
        # given by Eq. 12 in Esquivel+2003, for scales down to 2 pix.
        dv_sq = vel_disp**2 * \
            (2. / float(vel_field.shape[los_axis]))**(vel_struct_index)

        v_eff = np.sqrt(dv_sq + 2 * v_therm_sq).to(vel_field.unit)

        N_chan = int(np.ceil((v_eff / 5.).value))

    else:
        N_chan = int(np.ceil(del_v / chan_width))

    if verbose:
        print("Number of spectral channels {}".format(N_chan))
        print("Min velocity {}".format(v_min))
        print("Max velocity {}".format(v_max))
        print("Channel width {}".format(chan_width))

    if N_chan < 6:
        warn("<6 channels will be used ({} channels). Recommend increasing the"
             " number of channels. See Esquivel et al. (2003) and Chepurnov & "
             "Lazarian (2009).")

    if N_chan > max_chan:
        raise ValueError("Number of channels larger than set maximum"
                         " ({})".format(max_chan))

    # When computing the spectrum, we want the edges of the velocity channels
    # numpy in py27 isn't handling the units correctly in linspace.
    vel_edges = np.linspace(v_min.value, v_max.value, N_chan + 1) * v_min.unit

    vel_axis = 0.5 * (vel_edges[1:] + vel_edges[:-1])

    # Length of one pixel
    pix_scale = los_length.to(u.cm) / float(vel_field.shape[los_axis])

    shape = list(vel_field.shape)
    shape.pop(los_axis)

    spec_gen = ((vel_edges, vel_field[field_slice(y, x, los_axis)],
                 dens_field[field_slice(y, x, los_axis)],
                 v_therm_sq, pix_scale, y, x) for y, x in
                product(range(shape[0]), range(shape[1])))

    if threads == 1:
        spectra = list(map(_mapper, spec_gen))
    else:

        with Pool(threads) as pool:

            spectra = pool.map(_mapper, spec_gen)

    # Map spectra into a cube
    cube = np.empty(vel_axis.shape + tuple(shape)) * u.K

    for out in spectra:
        spec, y, x = out
        cube[:, y, x] = spec

    if return_hdu:
        header = create_cube_header(pixel_ang_scale, np.diff(vel_axis)[0],
                                    0.0 * u.arcsec, cube.shape, restfreq, u.K,
                                    v0=vel_axis[0])

        return fits.PrimaryHDU(cube.value, header)

    return cube, vel_axis


def _spectrum_maker(vel_edges, vel_slice, dens_slice, v_therm_sq, pix_scale,
                    y, x):
    '''
    Generate an optically-thin spectrum.
    '''

    spectrum = np.zeros_like(vel_edges.value[:-1]) * u.cm**-2 / (u.cm / u.s)

    # Make derivative of los velocity.

    dvdz = (np.gradient(vel_slice) / pix_scale).to(1 / u.s)

    v_cents = 0.5 * (vel_edges[1:] + vel_edges[:-1])

    for i, (vel_l, vel_h) in enumerate(zip(vel_edges[:-1], vel_edges[1:])):

        # Find locations of all contributions at this velocity
        contribs = np.logical_and(vel_slice > vel_l, vel_slice < vel_h)

        if not contribs.any():
            continue

        sigma = np.sqrt((dvdz[contribs] * pix_scale)**2 +
                        v_therm_sq).to(u.cm / u.s)

        # from matplotlib import pyplot as plt

        for jj, ni in enumerate(np.where(contribs)[0]):

            # Column density normalized by Gaussian integral
            # Output in units of u.cm**-2 / (u.cm / u.s)
            term1 = (dens_slice[ni] * pix_scale) / (SQRT_2PI * sigma[jj])

            term2 = ((v_cents - vel_slice[ni])**2 /
                     (2 * sigma[jj]**2)).to(u.dimensionless_unscaled)

            spectrum += (term1 * np.exp(-term2))

            # print(vel_slice[ni], sigma[jj])

            # print(dens_slice[ni] * pix_scale)
            # print(spectrum.sum() * (vel_h - vel_l))

            # plt.plot(v_cents, spectrum)
            # plt.draw()
            # input("?")
            # plt.clf()

    spectrum /= conv_to_K

    return spectrum, y, x


def _mapper(inps):
    return _spectrum_maker(*inps)


def field_slice(y, x, los_axis):

    los_slice = slice(None)

    spat_slices = [y, x]

    spat_slices.insert(los_axis, los_slice)

    return spat_slices
