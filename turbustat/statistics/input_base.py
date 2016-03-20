# Licensed under an MIT open source license - see LICENSE

from astropy.io.fits import PrimaryHDU
from spectral_cube import SpectralCube
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject


def input_data(data):
    '''
    Accept a variety of input data forms and return those expected by the
    various statistics.
    '''

    if isinstance(data, PrimaryHDU):
        return (data.data, data.header)
    elif isinstance(data, SpectralCube):
        return (data.filled_data[:].value, data.header)
    elif isinstance(data, LowerDimensionalObject):
        return (data.value, data.header)
    elif isinstance(data, tuple) or isinstance(data, list):
        if len(data) != 2:
            raise TypeError("Must have two items: data and the header.")
        return data
    else:
        raise TypeError("Input data is not of an accepted form:"
                        " astropy.io.fits.PrimaryHDU, SpectralCube,"
                        " spectral_cube.LowerDimensionalObject or a tuple or"
                        " list containing the data and header, in that order.")
