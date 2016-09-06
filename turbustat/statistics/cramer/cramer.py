# Licensed under an MIT open source license - see LICENSE


'''

Implementation of the Cramer Statistic

'''

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from ..threeD_to_twoD import _format_data
from ...io import input_data, common_types, threed_types


class Cramer_Distance(object):
    """
    Compute the Cramer distance between two data cubes. The data cubes
    are flattened spatially to give 2D objects. We clip off empty channels
    and keep only the top quartile in the remaining channels.

    Parameters
    ----------

    cube1 : %(dtypes)s
        First cube to compare.
    cube2 : %(dtypes)s
        Second cube to compare.
    noise_value1 : float, optional
        Noise level in the first cube.
    noise_value2 : float, optional
        Noise level in the second cube.
    data_format : str, optional
        Method to arange cube into 2D. Only 'intensity' is currently
        implemented.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, noise_value1=0.1,
                 noise_value2=0.1):
        super(Cramer_Distance, self).__init__()
        self.cube1 = input_data(cube1, no_header=True)
        self.cube2 = input_data(cube2, no_header=True)

        self.noise_value1 = noise_value1
        self.noise_value2 = noise_value2

        self.data_matrix1 = None
        self.data_matrix2 = None
        self.distance = None

    def format_data(self, data_format='intensity', seed=13024):
        '''
        Rearrange data into a 2D object using the given format.
        '''

        self.data_matrix1 = _format_data(self.cube1, data_format=data_format,
                                         noise_lim=self.noise_value1)
        self.data_matrix2 = _format_data(self.cube2, data_format=data_format,
                                         noise_lim=self.noise_value2)

        # Need to check if the same number of samples is taken
        samps1 = self.data_matrix1.shape[1]
        samps2 = self.data_matrix2.shape[1]

        if samps1 != samps2:

            # Set the seed due to the sampling
            np.random.seed(seed)

            if samps1 < samps2:

                new_data = np.empty((self.data_matrix2.shape[0], samps1))

                for i in range(self.data_matrix2.shape[0]):
                    new_data[i, :] = \
                        np.random.choice(self.data_matrix2[i, :], samps1,
                                         replace=False)

                self.data_matrix2 = new_data

            else:

                new_data = np.empty((self.data_matrix1.shape[0], samps2))

                for i in range(self.data_matrix1.shape[0]):
                    new_data[i, :] = \
                        np.random.choice(self.data_matrix1[i, :], samps2,
                                         replace=False)

                self.data_matrix1 = new_data

    def cramer_statistic(self, n_jobs=1):
        '''
        Applies the Cramer Statistic to the datasets.

        Parameters
        ----------

        n_jobs : int, optional
            Sets the number of cores to use to calculate
            pairwise distances
        '''
        # Adjust what we call n,m based on the larger dimension.
        # Then the looping below is valid.
        if self.data_matrix1.shape[0] >= self.data_matrix2.shape[0]:
            m = self.data_matrix1.shape[0]
            n = self.data_matrix2.shape[0]
            larger = self.data_matrix1
            smaller = self.data_matrix2
        else:
            n = self.data_matrix1.shape[0]
            m = self.data_matrix2.shape[0]
            larger = self.data_matrix2
            smaller = self.data_matrix1

        pairdist11 = pairwise_distances(
            larger, metric="euclidean", n_jobs=n_jobs)
        pairdist22 = pairwise_distances(
            smaller, metric="euclidean", n_jobs=n_jobs)
        pairdist12 = pairwise_distances(
            larger, smaller,
            metric="euclidean", n_jobs=n_jobs)

        term1 = 0.0
        term2 = 0.0
        term3 = 0.0
        for i in range(m):
            for j in range(n):
                term1 += pairdist12[i, j]
            for ii in range(m):
                term2 += pairdist11[i, ii]

            if i < n:
                for jj in range(n):
                    term3 += pairdist22[i, jj]

        m, n = float(m), float(n)

        term1 *= (1 / (m * n))
        term2 *= (1 / (2 * m ** 2.))
        term3 *= (1 / (2 * n ** 2.))

        self.distance = (m * n / (m + n)) * (term1 - term2 - term3)

    def distance_metric(self, n_jobs=1):
        '''

        This serves as a simple wrapper in order to remain with the coding
        convention used throughout the rest of this project.

        '''

        self.format_data()
        self.cramer_statistic(n_jobs=n_jobs)

        return self
