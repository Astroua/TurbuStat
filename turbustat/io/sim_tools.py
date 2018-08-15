# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

'''
Routines for using TurbuStat with simulated observations.
'''

import astropy.units as u
from astropy.io import fits


def create_fits_hdu(data, *header_args):
    '''
    Return a FITS hdu for a numpy array of data.
    '''

    if data.ndim == 2:
        return fits.PrimaryHDU(data,
                               header=create_image_header(*header_args))
    elif data.ndim == 3:
        return fits.PrimaryHDU(data,
                               header=create_cube_header(*header_args))
    else:
        raise ValueError("data must be 2D or 3D.")


def create_cube_header(pixel_scale, spec_pixel_scale, beamfwhm, imshape,
                       restfreq, bunit, v0=0 * u.km / u.s):
    '''
    Create a basic FITS header for a PPV cube.

    Only frequency and radio velocity are currently supported for the spectral
    dimension type.

    Adapted from: https://github.com/radio-astro-tools/uvcombine/blob/master/uvcombine/tests/utils.py

    Parameters
    ----------
    pixel_scale : `~astropy.units.Quantity`
        Angular scale of one pixel
    spec_pixel_scale : `~astropy.units.Quantity`
        Spectral size of one pixel. Currently must be in equivalent radio
        velocity units.
    beamfwhm : `~astropy.units.Quantity`
        Angular size for a circular Gaussian beam.
    imshape : tuple
        Shape of the data array.
    restfreq : `~astropy.units.Quantity`
        Rest frequency of the spectral line.
    bunit : `~astropy.units.Unit`
        Unit of intensity.
    v0 : `~astropy.units.Unit`, optional
        The value of the spectral axis at the first pixel in the cube.

    Returns
    -------
    header : fits.Header
        FITS Header.
    '''

    vel_type = "FREQ" if spec_pixel_scale.unit.is_equivalent(u.Hz) else \
        "VRAD"

    header = {'CDELT1': -(pixel_scale).to(u.deg).value,
              'CDELT2': (pixel_scale).to(u.deg).value,
              'BMAJ': beamfwhm.to(u.deg).value,
              'BMIN': beamfwhm.to(u.deg).value,
              'BPA': 0.0,
              'CRPIX1': imshape[1] / 2.,
              'CRPIX2': imshape[2] / 2.,
              'CRVAL1': 0.0,
              'CRVAL2': 0.0,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CUNIT1': 'deg',
              'CUNIT2': 'deg',
              'CRVAL3': v0.to(spec_pixel_scale.unit).value,
              'CUNIT3': spec_pixel_scale.unit.to_string(),
              'CDELT3': spec_pixel_scale.value,
              'CRPIX3': 1,
              'CTYPE3': vel_type,
              'RESTFRQ': restfreq.to(u.Hz).value,
              'BUNIT': bunit.to_string(),
              }

    return fits.Header(header)


def create_image_header(pixel_scale, beamfwhm, imshape,
                        restfreq, bunit):
    '''
    Create a basic FITS header for an image.

    Adapted from: https://github.com/radio-astro-tools/uvcombine/blob/master/uvcombine/tests/utils.py

    Parameters
    ----------
    pixel_scale : `~astropy.units.Quantity`
        Angular scale of one pixel
    beamfwhm : `~astropy.units.Quantity`
        Angular size for a circular Gaussian beam.
    imshape : tuple
        Shape of the data array.
    restfreq : `~astropy.units.Quantity`
        Rest frequency of the spectral line.
    bunit : `~astropy.units.Unit`
        Unit of intensity.

    Returns
    -------
    header : fits.Header
        FITS Header.
    '''

    header = {'CDELT1': -(pixel_scale).to(u.deg).value,
              'CDELT2': (pixel_scale).to(u.deg).value,
              'BMAJ': beamfwhm.to(u.deg).value,
              'BMIN': beamfwhm.to(u.deg).value,
              'BPA': 0.0,
              'CRPIX1': imshape[0] / 2.,
              'CRPIX2': imshape[1] / 2.,
              'CRVAL1': 0.0,
              'CRVAL2': 0.0,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CUNIT1': 'deg',
              'CUNIT2': 'deg',
              'RESTFRQ': restfreq.to(u.Hz).value,
              'BUNIT': bunit.to_string(),
              }

    return fits.Header(header)
