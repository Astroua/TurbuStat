
from astropy.io import fits
import numpy as np

from ..io import input_data


class BaseStatisticMixIn(object):
    """
    Common properties to all statistics
    """

    # Disable this flag when a statistic does not need a header
    need_header_flag = True

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

        if not isinstance(values, np.ndarray):
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
