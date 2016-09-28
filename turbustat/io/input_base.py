# Licensed under an MIT open source license - see LICENSE

from astropy.io.fits import PrimaryHDU
from spectral_cube import SpectralCube
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject
import numpy as np


common_types = ["numpy.ndarray", "astropy.io.fits.PrimaryHDU"]
twod_types = ["spectral_cube.LowerDimensionalObject"]
threed_types = ["SpectralCube"]


def input_data(data, no_header=False):
    '''
    Accept a variety of input data forms and return those expected by the
    various statistics.

    Parameters
    ----------
    data : astropy.io.fits.PrimaryHDU, SpectralCube,
           spectral_cube.LowerDimensionalObject, np.ndarray or a tuple/list
           with the data and the header
        Data to be used with a given statistic or distance metric. no_header
        must be enabled when passing only an array in.
    no_header : bool, optional
        When enabled, returns only the data without the header.

    Returns
    -------
    ouput_data : tuple or np.ndarray
        A tuple containing the data and the header. Or an array when no_header
        is enabled.
    '''

    if isinstance(data, PrimaryHDU):
        output_data = [data.data, data.header]
    elif isinstance(data, SpectralCube):
        output_data = [data.filled_data[:].value, data.header]
    elif isinstance(data, LowerDimensionalObject):
        output_data = [data.value, data.header]
    elif isinstance(data, tuple) or isinstance(data, list):
        if len(data) != 2:
            raise TypeError("Must have two items: data and the header.")
        output_data = data
    elif isinstance(data, np.ndarray):
        if not no_header:
            raise TypeError("no_header must be enabled when giving data"
                            " without a header.")
        output_data = [data]
    else:
        raise TypeError("Input data is not of an accepted form:"
                        " astropy.io.fits.PrimaryHDU, SpectralCube,"
                        " spectral_cube.LowerDimensionalObject or a tuple or"
                        " list containing the data and header, in that order.")

    if no_header:
        return output_data[0]

    return output_data
