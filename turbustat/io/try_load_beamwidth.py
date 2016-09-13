
try:
    from radio_beam import Beam
    RADIO_BEAM_INSTALL = False
except ImportError:
    RADIO_BEAM_INSTALL = True

from astropy.io import fits
from astropy import units as u


def find_beam_width(hdr):
    '''
    Find the beam width from a FITS header. If radio_beam is installed, use it
    since it is more robust at loading from headers.

    Otherwise, check for BMAJ and fail if it isn't found.

    '''
    if RADIO_BEAM_INSTALL:
        beam = Beam.from_fits_header(hdr)
        beamwidth = beam.major.to(u.deg)
    else:
        if not isinstance(hdr, fits.Header):
            raise TypeError("Header is not a FITS header.")

        if "BMAJ" in hdr:
            beamwidth = hdr["BMAJ"] * u.deg
        else:
            raise ValueError("Cannot find 'BMAJ' in the header. Try installing"
                             " the `radio_beam` package for loading header"
                             " information.")

    return beamwidth
