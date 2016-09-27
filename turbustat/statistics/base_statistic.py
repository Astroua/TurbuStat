
from astropy.io import fits
import astropy.units as u
import numpy as np
from astropy.wcs import WCS

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
    def _wcs(self):
        if not hasattr(self, "_header"):
            raise AttributeError("No header was found.")

        return WCS(self.header)

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
        if not hasattr(self, "_header"):
            raise AttributeError("No header has not been given.")

        return [(u.pix, u.deg, lambda x: x * float(self.ang_size.value),
                lambda x: x / float(self.ang_size.value))]

    @property
    def ang_size(self):
        if not hasattr(self, "_header"):
            raise AttributeError("No header has not been given.")

        return np.abs(self.header["CDELT2"]) * u.Unit(self._wcs.wcs.cunit[1])

    def to_pixel(self, value):
        '''
        Convert from angular to pixel scale.
        '''

        if not isinstance(value, u.Quantity):
            raise TypeError("value must be an astropy Quantity object.")

        return value.to(u.pix, equivalencies=self.angular_equiv)

    @property
    def distance(self):
        if not hasattr(self, "_header"):
            raise AttributeError("No header has not been given. Cannot make"
                                 " use of distance.")

        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return self._distance

    @distance.setter
    def distance(self, value):
        '''
        Value must be a quantity with a valid distance unit. Will keep the
        units given.
        '''

        if not isinstance(value, u.Quantity):
            raise TypeError("Value for distance must an astropy Quantity.")

        if not value.unit.is_equivalent(u.pc):
            raise u.UnitConversionError("Given unit ({}) is not a valid unit"
                                        " of distance.".format(value.unit))

        if not value.isscalar:
            raise TypeError("Distance must be a scalar quantity.")

        self._distance = value

    @property
    def distance_size(self):
        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return (self.ang_size *
                self.distance).to(self.distance.unit,
                                  equivalencies=u.dimensionless_angles())

    @property
    def distance_equiv(self):
        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return [(u.pix, self.distance.unit,
                lambda x: x * float(self.distance_size.value),
                lambda x: x / float(self.distance_size.value))]

    def to_physical(self, value):
        if not hasattr(self, "_distance"):
            raise AttributeError("No distance has not been given.")

        return value.to(self.distance.unit, equivalencies=self.distance_equiv)

    def has_spectral(self, raise_error=False):
        '''
        Test whether there is a spectral axis.
        '''
        axtypes = self._wcs.get_axis_types()

        types = [a['coordinate_type'] for a in axtypes]

        if 'spectral' not in types:
            if raise_error:
                raise ValueError("Header does not have spectral axis.")
            return False

        return True

    @property
    def spectral_size(self):

        self.has_spectral(raise_error=True)

        spec = self._wcs.wcs.spec
        return np.abs(self._wcs.wcs.cdelt[spec]) * \
            u.Unit(self._wcs.wcs.cunit[spec])

    @property
    def spectral_equiv(self):
        return [(u.pix, self.spectral_size.unit,
                lambda x: x * float(self.spectral_size.value),
                lambda x: x / float(self.spectral_size.value))]
