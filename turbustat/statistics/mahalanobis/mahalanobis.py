
import numpy as np
from scipy.spatial.distance import mahalanobis

from ..threeD_to_twoD import _format_data
from ..mantel import mantel_test


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

    def compute_distmat(self):
        '''
        Compute the Mahalanobis distance matrix.
        '''

        channels = self.data_matrix.shape[0]

        self.distance_matrix = np.zeros((channels, channels))

        for i in range(channels):
            for j in range(i):
                self.distance_matrix[i, j] = \
                    mahala_fcn(self.data_matrix[i, :], self.data_matrix[j, :])

        # Add in the upper triangle
        self.distance_matrix = self.distance_matrix + self.distance_matrix.T

        return self

    def run(self, verbose=False, *args):
        '''
        Run all computations.
        '''
        self.format_data(*args)
        self.compute_distmat()

        if verbose:
            import matplotlib.pyplot as p

            p.imshow(self.distance_matrix, interpolation=None)
            p.colorbar()
            p.show()

        return self


class Mahalanobis_Distance(object):

    """docstring for Mahalanobis_Distance"""

    def __init__(self, cube1, cube2):
        super(Mahalanobis_Distance, self).__init__()

        self.mahala1 = Mahalanobis(cube1)
        self.mahala2 = Mahalanobis(cube2)

    def compute_distmats(self, data_format='spectra', *args):
        '''
        Create a 2D representation of the data. args are passed to
        _format_data.
        '''

        self.mahala1.format_data(data_format=data_format, *args)
        self.mahala2.format_data(data_format=data_format, *args)

        return self

    def distance_metric(self, correlation='pearson', verbose=False):
        '''

        This serves as a simple wrapper in order to remain with the coding
        convention used throughout the rest of this project.

        '''

        self.distance, self.pval = \
            mantel_test(self.mahala1.distance_matrix,
                        self.mahala2.distance_matrix,
                        corr_func=correlation)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(121)
            p.title('Results: Distance - '+str(self.distance))
            p.imshow(self.mahala1.distance_matrix, interpolation='nearest')
            p.subplot(122)
            p.title('Results: P-value - '+str(self.pval))
            p.imshow(self.mahala2.distance_matrix, interpolation='nearest')
            p.show()

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

    val = mahalanobis(x, y, icov)

    return np.sqrt(val)
