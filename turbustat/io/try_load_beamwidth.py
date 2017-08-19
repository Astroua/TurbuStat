# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

try:
    from radio_beam import Beam
    RADIO_BEAM_INSTALL = True
except ImportError:
    RADIO_BEAM_INSTALL = False

from astropy.io import fits
from astropy import units as u
from warnings import warn


def find_beam_properties(hdr):
    '''
    Try to read beam properties from a header. Uses radio_beam when installed.

    Parameters
    ----------
    hdr : `~astropy.io.fits.Header`
        FITS header.

    Returns
    -------
    bmaj : `~astropy.units.Quantity`
        Major axis of the beam in degrees.
    bmin : `~astropy.units.Quantity`
        Minor axis of the beam in degrees. If this cannot be read from the
        header, assumes `bmaj=bmin`.
    bpa : `~astropy.units.Quantity`
        Position angle of the major axis. If this cannot read from the
        header, assumes an angle of 0 deg.
    '''

    if RADIO_BEAM_INSTALL:
        beam = Beam.from_fits_header(hdr)
        bmaj = beam.major.to(u.deg)
        bmin = beam.minor.to(u.deg)
        bpa = beam.pa.to(u.deg)
    else:
        if not isinstance(hdr, fits.Header):
            raise TypeError("Header is not a FITS header.")

        if "BMAJ" in hdr:
            bmaj = hdr["BMAJ"] * u.deg
        else:
            raise ValueError("Cannot find 'BMAJ' in the header. Try installing"
                             " the `radio_beam` package for loading header"
                             " information.")

        if "BMIN" in hdr:
            bmin = hdr["BMIN"] * u.deg
        else:
            warn("Cannot find 'BMIN' in the header. Assuming circular beam.")
            bmin = bmaj

        if "BPA" in hdr:
            bpa = hdr["BPA"] * u.deg
        else:
            warn("Cannot find 'BPA' in the header. Assuming PA of 0.")
            bpa = 0 * u.deg

    return bmaj, bmin, bpa


def find_beam_width(hdr):
    '''
    Find the beam width from a FITS header. If radio_beam is installed, use it
    since it is more robust at loading from headers.

    Otherwise, check for BMAJ and fail if it isn't found.

    '''
    bmaj, bmin, bpa = find_beam_properties(hdr)

    return bmaj
