# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

from astropy.io.fits.hdu.image import _ImageBaseHDU
from astropy.io import fits
import numpy as np

try:
    from spectral_cube.version import version as sc_version
    from distutils.version import LooseVersion
    if LooseVersion(sc_version) < LooseVersion("0.4.4"):
        raise ValueError("turbustat requires spectral-cube version 0.4.4."
                         " Found version {}".format(sc_version))
    from spectral_cube import SpectralCube
    from spectral_cube.lower_dimensional_structures import LowerDimensionalObject
    HAS_SC = True
except ImportError:
    HAS_SC = False


common_types = ["numpy.ndarray", "astropy.io.fits.PrimaryHDU",
                "astropy.io.fits.ImageHDU"]
twod_types = ["spectral_cube.Projection", "spectral_cube.Slice"]
threed_types = ["SpectralCube"]


def input_data(data, no_header=False, need_copy=False):
    '''
    Accept a variety of input data forms and return those expected by the
    various statistics.

    Parameters
    ----------
    data : astropy.io.fits.PrimaryHDU, spectral_cube.SpectralCube,
           spectral_cube.Projection, spectral_cube.Slice, np.ndarray or a
           tuple/listwith the data and the header
        Data to be used with a given statistic or distance metric. no_header
        must be enabled when passing only an array in.
    no_header : bool, optional
        When enabled, returns only the data without the header.
    need_copy : bool, optional
        Return a copy of the data when enabled.

    Returns
    -------
    ouput_data : tuple or np.ndarray
        A tuple containing the data and the header. Or an array when no_header
        is enabled.
    '''

    def make_copy(data, need_copy):
        if need_copy:
            return data.copy()
        else:
            return data

    if HAS_SC:
        sc_def = False
        if isinstance(data, SpectralCube):
            output_data = [make_copy(data.filled_data[:].value, need_copy),
                           data.header]
            sc_def = True
        elif isinstance(data, LowerDimensionalObject):
            output_data = [make_copy(data.value, need_copy), data.header]
            sc_def = True
    else:
        sc_def = False

    if not sc_def:
        if isinstance(data, _ImageBaseHDU):
            output_data = [make_copy(data.data, need_copy), data.header]
        elif isinstance(data, tuple) or isinstance(data, list):
            if len(data) != 2:
                raise TypeError("Must have two items: data and the header.")
            output_data = data

        elif isinstance(data, np.ndarray):
            if not no_header:
                raise TypeError("no_header must be enabled when giving data"
                                " without a header.")
            output_data = [make_copy(data, need_copy)]
        else:
            raise TypeError("Input data is not of an accepted form:"
                            " astropy.io.fits.PrimaryHDU, "
                            " astropy.io.fits.ImageHDU,"
                            " spectral_cube.SpectralCube,"
                            " spectral_cube.LowerDimensionalObject"
                            " or a tuple or list containing the data and"
                            " header, in that order.")

    if no_header:
        return output_data[0]

    return output_data


def to_spectral_cube(data, header):
    '''
    Convert the output from input_data into a SpectralCube.
    '''

    if not HAS_SC:
        raise ValueError("spectral-cube needs to be installed.")

    hdu = fits.PrimaryHDU(data, header)

    return SpectralCube.read(hdu)
