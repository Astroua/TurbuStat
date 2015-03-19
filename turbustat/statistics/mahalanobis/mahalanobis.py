
import numpy as np

from ..threeD_to_twoD import _format_data


class Mahalanobis(object):
    """docstring for Mahalanobis"""
    def __init__(self, cube):
        super(Mahalanobis, self).__init__()
        self.cube = cube

    def format_data(self, data_format='spectra', *args):
        '''
        Create a 2D representation of the data. args are passed to
        _format_data.
        '''

        self.data_matrix = _format_data(self.cube, data_format=data_format,
                                        *args)

        return self


class Mahalanobis_Distance(object):

    """docstring for Mahalanobis_Distance"""

    def __init__(self, cube1, cube2, noise_value1=0.1,
                 noise_value2=0.1, data_format="intensity"):
        super(Mahalanobis_Distance, self).__init__()
        self.cube1 = cube1
        self.cube2 = cube2
        self.data_format = data_format

        self.noise_value1 = noise_value1
        self.noise_value2 = noise_value2

        self.data_matrix1 = None
        self.data_matrix2 = None
        self.distance = None

    def format_data(self, data_format=None, num_spec=1000):
        '''
        Rearrange data into a 2D object using the given format.
        '''

        if data_format is not None:
            self.data_format = data_format

        if self.data_format is "spectra":
            if num_spec is None:
                raise ValueError('Must specify num_spec for data format',
                                 'spectra.')

            # Find the brightest spectra in the cube
            mom0 = np.nansum(self.cube1)

            bright_spectra = \
                np.argpartition(mom0.ravel(), -num_spec)[-num_spec:]

            x = np.empty((num_spec,))
            y = np.empty((num_spec,))

            for i in range(num_spec):
                x[i] = bright_spectra[i] / self.cube1.shape[1]
                y[i] = bright_spectra[i] % self.cube1.shape[2]

            self.data_matrix1 = self.cube1[:, x, y]

        elif self.data_format is "intensity":
            self.data_matrix1 = intensity_data(self.cube1,
                                               noise_lim=self.noise_value1)
            self.data_matrix2 = intensity_data(self.cube2,
                                               noise_lim=self.noise_value2)

        else:
            raise NameError(
                "data_format must be either 'spectra' or 'intensity'.")

        return self

    def mahalanobis_statistic(self, n_jobs=1):
        '''

        '''
        # Adjust what we call n,m based on the larger dimension.
        # Then the looping below is valid.
        if self.data_matrix1.shape[1] >= self.data_matrix2.shape[1]:
            m = self.data_matrix1.shape[1]
            n = self.data_matrix2.shape[1]
        else:
            n = self.data_matrix1.shape[1]
            m = self.data_matrix2.shape[1]

        term1 = 0.0
        term2 = 0.0
        term3 = 0.0

        for i in range(m):
            for j in range(n):
                term1 += mahala_fcn(self.data_matrix1[:, i],
                                    self.data_matrix2[:, j])
            for ii in range(m):
                term2 += mahala_fcn(self.data_matrix1[:, i],
                                    self.data_matrix1[:, ii])

            if i <= n:
                for jj in range(n):
                    term3 += mahala_fcn(
                        self.data_matrix2[:, i], self.data_matrix2[:, jj])

        m, n = float(m), float(n)

        term1 *= (1 / (m * n))
        term2 *= (1 / (2 * m**2.))
        term3 *= (1 / (2 * n**2.))

        self.distance = (m * n / (m + n)) * (term1 - term2 - term3)

        return self

    def distance_metric(self, n_jobs=1):
        '''

        This serves as a simple wrapper in order to remain with the coding
        convention used throughout the rest of this project.

        '''

        self.format_data()
        self.mahalanobis_statistic(n_jobs=n_jobs)

        return self


def mahala_fcn(x, y):
    '''

    Parameters
    ----------

    x - numpy.ndarray
        A 1D array

    y - numpy.ndarray
        A 1D array

    '''

    cov = np.cov(zip(x, y))
    try:
        icov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        icov = np.linalg.inv(cov + np.eye(cov.shape[0], cov.shape[1], k=1e-3))

    diff = x - y
    val = np.dot(diff.T, np.dot(icov, diff))

    return np.sqrt(diff)
