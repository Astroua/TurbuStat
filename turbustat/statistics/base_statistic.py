
from astropy.io import fits
import astropy.units as u
import numpy as np

from ..io import input_data


class BaseStatisticMixIn(object):
    """
    Common properties to all statistics
    """

    # Disable this flag when a statistic does not need a header
    need_header_flag = True

    # Disable this when the data property will not be used.
    no_data_flag = False

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, input_hdr):

        if not self.need_header_flag:
            input_hdr = None

        elif not isinstance(input_hdr, fits.header.Header):
            raise TypeError("The header must be a"
                            " astropy.io.fits.header.Header.")

        self._header = input_hdr

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):

        if self.no_data_flag:
            values = None

        elif not isinstance(values, np.ndarray):
            raise TypeError("Data is not a numpy array.")

        self._data = values

    def input_data_header(self, data, header):
        '''
        Check if the header is given separately from the data type.
        '''

        if header is not None:
            self.data = input_data(data, no_header=True)
            self.header = header
        else:
            self.data, self.header = input_data(data)

    @property
    def angular_equiv(self):
        return [(u.pix, u.deg, lambda x: x * float(self.ang_size.value),
                lambda x: x / float(self.ang_size.value))]

    @property
    def ang_size(self):
        return np.abs(self.header["CDELT2"]) * u.deg

    def to_pixel(self, value):
        '''
        Convert from angular to pixel scale.
        '''

        if not isinstance(value, u.Quantity):
            raise TypeError("value must be an astropy Quantity object.")

        return value.to(u.pix, equivalencies=self.angular_equiv)
