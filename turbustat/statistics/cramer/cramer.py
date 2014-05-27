
'''

Implementation of the Cramer Statistic

'''

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Cramer_Distance(object):

    """docstring for Cramer_Distance"""

    def __init__(self, cube1, cube2, data_format="intensity"):
        super(Cramer_Distance, self).__init__()
        self.cube1 = cube1
        self.cube2 = cube2
        self.data_format = data_format

        self.data_matrix1 = None
        self.data_matrix2 = None
        self.distance = None

    def format_data(self, data_format=None):

        if data_format is not None:
            self.data_format = data_format

        if self.data_format == "spectra":
            raise NotImplementedError("")

        elif self.data_format == "intensity":
            self.data_matrix1 = intensity_data(self.cube1)
            self.data_matrix2 = intensity_data(self.cube2)

        else:
            raise NameError(
                "data_format must be either 'spectra' or 'intensity'.")

        return self

    def cramer_statistic(self, n_jobs=1):
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

        pairdist11 = pairwise_distances(
            self.data_matrix1.T, metric="euclidean", n_jobs=n_jobs)
        pairdist22 = pairwise_distances(
            self.data_matrix2.T, metric="euclidean", n_jobs=n_jobs)
        pairdist12 = pairwise_distances(
            self.data_matrix1.T, self.data_matrix2.T,
            metric="euclidean", n_jobs=n_jobs)

        for i in range(m):
            for j in range(n):
                term1 += pairdist12[i, j]
            for ii in range(m):
                term2 += pairdist11[i, ii]

            if i <= n:
                for jj in range(n):
                    term3 += pairdist22[i, jj]

        m, n = float(m), float(n)

        term1 *= (1 / (m * n))
        term2 *= (1 / (2 * m ** 2.))
        term3 *= (1 / (2 * n ** 2.))

        self.distance = (m * n / (m + n)) * (term1 - term2 - term3)

        return self

    def distance_metric(self, n_jobs=1):
        '''

        This serves as a simple wrapper in order to remain with the coding
        convention used throughout the rest of this project.

        '''

        self.format_data()
        self.cramer_statistic(n_jobs=n_jobs)

        return self


def intensity_data(cube, p=0.25):
    '''
    '''
    vec_length = int(round(p * cube.shape[1] * cube.shape[2]))
    intensity_vecs = np.empty((cube.shape[0], vec_length))
    cube[np.isnan(cube)] = 0.0
    for dv in range(cube.shape[0]):
        vel_vec = cube[dv, :, :].ravel()
        vel_vec.sort()
        if len(vel_vec) < vec_length:
            diff = vec_length - len(vel_vec)
            vel_vec = np.append(vel_vec, [0.0] * diff)

        # Return the normalized, shortened vector
        maxval = np.max(vel_vec[:vec_length])
        if maxval != 0.0:
            intensity_vecs[dv, :] = vel_vec[:vec_length]/maxval
        else:
            intensity_vecs[dv, :] = vel_vec[:vec_length]  # Vector of zeros
    return intensity_vecs
