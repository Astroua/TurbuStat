
import numpy as np
import astropy.constants as co
import astropy.units as u
from itertools import product
from warnings import warn
from multiprocessing import Pool
from astropy.utils.console import ProgressBar
from astropy.io import fits

from ..io.sim_tools import create_cube_header
from .spectrum import generate_spectrum

SQRT_2PI = np.sqrt(2 * np.pi)
conv_to_K = 1.823e13 * u.cm**-2 / (u.K * u.cm / u.s)


def make_ppv(vel_field, dens_field, los_axis=0,
             m=1.4 * co.m_p, T=100 * u.K, los_length=1 * u.pc,
             vel_disp=None, chan_width=None, v_min=None, v_max=None,
             threads=1, max_chan=1000,
             vel_struct_index=0.5, verbose=False,
             return_hdu=True, pixel_ang_scale=1 * u.arcmin,
             restfreq=1.42 * u.GHz):
    '''
    Generate a mock, optically-thin HI PPV cube from a given velocity and
    density field. Currently, the conversion to K assumes the 21-cm column
    density conversion.

    Parameters
    ----------
    vel_field : `~astropy.units.Quantity`
        Three-dimensional isotropic, velocity field. Must have units of
        velocity.
    dens_field : `~astropy.units.Quantity`
        Three-dimensional isotropic, velocity field. Must have units of number
        density.
    los_axis : int, optional
        The axis used to produce the PPV cube.
    m : `~astropy.units.Quantity`, optional
        Average particle mass. Defaults to 1.4 time the proton mass,
        appropriate for the neutral ISM at solar metallicity. Used to
        calculate the thermal line width.
    T : `~astropy.units.Quantity`, optional
        Temperature of the gas. Defaults to 100 K. Used to
        calculate the thermal line width.
    los_length : `~astropy.units.Quantity`, optional
        Set the total physical length of the density and velocity fields
        along the line-of-sight. Defaults to 1 pc.
    vel_disp : `~astropy.units.Quantity`, optional
        Velocity dispersion of the 3D velocity field. When none is given,
        the mean of the projected standard deviation of the velocity field
        along `los_axis` is used. Used to set the extent of the velocity
        channels and estimate the channel widths, when not given.
    chan_width : `~astropy.units.Quantity`, optional
        Width of a velocity channel. When the width is not given, an
        estimate is made based on Eq. 12 from
        `Esquivel+2003 <https://ui.adsabs.harvard.edu/#abs/2003MNRAS.342..325E/abstract>`_.
    v_min : `~astropy.units.Quantity`, optional
        Minimum velocity channel. Set to `vel_field - 4 * v_lim`, where
        `v_lim = sqrt(vel_disp**2 + v_therm**2)`, when a limit is not given.
    v_max : `~astropy.units.Quantity`, optional
        Maximum velocity channel. Set to `vel_field + 4 * v_lim`, where
        `v_lim = sqrt(vel_disp**2 + v_therm**2)`, when a limit is not given.
    threads : int, optional
        Number of cores to run on. Defaults to 1.
    max_chan : int, optional
        Sets an upper limit on the number of velocity channels (default of
        1000) to avoid using excessive amounts of memory. If the number of
        channels exceeds this limit, a `ValueError` is raised.
    vel_struct_index : float, optional
        Index of the velocity field. Used when automatically determining
        the channel width. Defaults to 0.5.
    verbose : bool, optional
        Print out the min and max velocity extents and the channel width
        prior to computing the cube.
    return_hdu : bool, optional
        Return the cube as a FITS HDU. Enabled by default.
    pixel_ang_scale : `~astropy.units.Quantity`, optional
        Specify the angular scale of one spatial pixel to set values in the
        FITS header. Defaults to 1 arcmin.
    restfreq : `~astropy.units.Quantity`, optional
        Rest frequency of the spectral line passed to the FITS header.
        Defaults to 1.42 GHz, roughly the 21-cm HI rest frequency.

    Returns
    -------
    cube : `~astropy.units.Quantity` or `~astropy.io.fits.PrimaryHDU`
        The PPV cube as an array (`return_hdu=False`) or a FITS HDU
        (`return_hdu=True`).
    vel_axis : `~astropy.units.Quantity`
        When `return_hdu=False` is returned, the values for the velocity axis
        are returned.
    '''

    # Densities better be postive
    if (dens_field.value < 0.).any():
        raise ValueError("The density field contains negative values.")

    v_therm_sq = (co.k_B * T / m).to(vel_field.unit**2)

    # Estimate the velocity dispersion when not given.
    # Use the
    if vel_disp is None:
        vel_disp = np.std(vel_field, axis=los_axis).mean()

    # Make a line width estimate with thermal broadening for setting velocity
    # extent
    v_lim = np.sqrt(vel_disp**2 + v_therm_sq)

    # Setup velocity axis
    if v_min is None:
        v_min = vel_field.min() - 4 * v_lim
    else:
        if not v_min.unit.is_equivalent(u.km / u.s):

            raise u.UnitsError("v_min must be given in velocity units.")
    if v_max is None:
        v_max = vel_field.max() + 4 * v_lim
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
        # Note that the equation in the paper has a negative sign in the sqrt
        # that should be a +.

        # Take the effective width to be ~1/2 of the effective channel width
        # given by Eq. 12 in Esquivel+2003, for scales down to 2 pix.
        dv_sq = vel_disp**2 * \
            (2. / float(vel_field.shape[los_axis]))**(vel_struct_index)

        v_eff = np.sqrt(dv_sq + 2 * v_therm_sq).to(vel_field.unit)

        N_chan = int(np.ceil(del_v / (v_eff / 2.)).value)

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

    v_cents = (0.5 * (vel_edges[1:] + vel_edges[:-1])).to(u.cm / u.s)

    spectrum = generate_spectrum(vel_slice.to(u.cm / u.s).value,
                                 dens_slice.to(u.cm**-3).value,
                                 vel_edges.to(u.cm / u.s).value,
                                 v_cents.value,
                                 dvdz.value,
                                 v_therm_sq.to(u.cm**2 / u.s**2).value,
                                 pix_scale.to(u.cm).value)

    spectrum = spectrum * u.cm**-2 / (u.cm / u.s)

    spectrum /= conv_to_K

    return spectrum, y, x


def _mapper(inps):
    '''
    Use with `multiprocessing.Pool.map`.
    '''
    return _spectrum_maker(*inps)


def field_slice(y, x, los_axis):
    '''
    Slice out spatial slices of a 3D field without the axis along
    the observer's LOS.
    '''

    los_slice = slice(None)

    spat_slices = [y, x]

    spat_slices.insert(los_axis, los_slice)

    return spat_slices
