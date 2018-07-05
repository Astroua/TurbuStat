
import numpy as np
import astropy.constants as co
import astropy.units as u
from itertools import product
from warnings import warn
from multiprocessing import Pool
from astropy.utils.console import ProgressBar

from .gen_field import make_3dfield


SQRT_2PI = np.sqrt(2 * np.pi)


def generate_velocity_field():
    pass


def generate_density_field():
    pass


def make_ppv(vel_field, dens_field, los_axis=0,
             m=1.4 * co.m_p, T=4000 * u.K, los_length=1 * u.pc,
             vel_disp=None, threads=1):
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
    v_min = vel_field.min() - 4 * v_therm
    v_max = vel_field.max() + 4 * v_therm

    N_chan = int(np.ceil((v_max - v_min) / vel_disp))

    # Thin channel criteria from Esquivel+2003 requires
    # Delta_v / vel_disp > 5~6
    # Stricter requirements for VCS from Chepurnov & Lazarian 2009 are not
    # used here to reduce computational requirements.
    if N_chan < 6:
        warn("<6 channels will be used ({} channels). Recommend increasing the"
             " number of channels. See Esquivel et al. (2003) and Chepurnov & "
             "Lazarian (2009).")

    # When computing the spectrum, we want the edges of the velocity channels
    vel_edges = np.linspace(v_min, v_max, N_chan + 1)

    vel_axis = 0.5 * (vel_edges[1:] + vel_edges[:-1])

    # Length of one pixel
    pix_scale = los_length.to(u.cm) / float(vel_field.shape[los_axis])

    shape = [*vel_field.shape]
    shape.pop(los_axis)

    spec_gen = ((vel_edges, vel_field[field_slice(y, x, los_axis)],
                 np.ones_like(vel_field[field_slice(y, x, los_axis)].value) * u.cm**-3,
                 v_therm_sq, pix_scale, y, x) for y, x in
                product(range(shape[0]), range(shape[1])))

    if threads == 1:
        spectra = list(map(_mapper, spec_gen))
    else:
        pool = Pool(threads)

        spectra = pool.map(_mapper, spec_gen)

        pool.join()
        pool.close()

    # Map spectra into a cube
    cube = np.empty(vel_axis.shape + tuple(shape)) * u.K

    for out in spectra:
        spec, y, x = out
        cube[:, y, x] = spec

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

        conv_to_K = 1.823e13 * u.cm**-2 / (u.K * u.cm / u.s)

    spectrum /= conv_to_K

    return spectrum, y, x


def _mapper(inps):
    return _spectrum_maker(*inps)


def field_slice(y, x, los_axis):

    los_slice = slice(None)

    spat_slices = [y, x]

    spat_slices.insert(los_axis, los_slice)

    return spat_slices
