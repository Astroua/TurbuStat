
'''

Implementation of the Mahalanobis Distance as a metric between two data cubes
Function has been
http://nbviewer.ipython.org/gist/kevindavenport/7771325

'''

import numpy as np


class Mahalanobis_Distance(object):
    """docstring for Mahalanobis_Distance"""
    def __init__(self, cube1, cube2, data_format="intensity"):
        super(Mahalanobis_Distance, self).__init__()
        self.cube1 = cube1
        self.cube2 = cube2
        self.data_format = data_format

        self.data_matrix1 = None
        self.data_matrix2 = None
        self.distance = None

    def format_data(self, data_format=None):

        if data_format is not None:
            self.data_format = data_format

        if self.data_format=="spectra":
            raise NotImplementedError("")

        elif self.data_format=="intensity":
            self.data_matrix1 = intensity_data(self.cube1)
            self.data_matrix2 = intensity_data(self.cube2)

        else:
            raise NameError("data_format must be either 'spectra' or 'intensity'.")

        return self

    def mahalanobis_statistic(self, n_jobs=1):
        '''

        '''
        ## Adjust what we call n,m based on the larger dimension.
        ## Then the looping below is valid.
        if self.data_matrix1.shape[1]>=self.data_matrix2.shape[1]:
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
                term1 += mahala_fcn(self.data_matrix1[:,i], self.data_matrix2[:,j])
            for ii in range(m):
                term2 += mahala_fcn(self.data_matrix1[:,i], self.data_matrix1[:,ii])

            if i<=n:
                for jj in range(n):
                    term3 += mahala_fcn(self.data_matrix2[:,i], self.data_matrix2[:,jj])

        m, n = float(m), float(n)

        term1 *= (1/(m*n))
        term2 *= (1/(2*m**2.))
        term3 *= (1/(2*n**2.))

        self.distance = (m*n/(m+n)) * (term1 - term2 - term3)

        return self

    def distance_metric(self, n_jobs=1):
        '''

        This serves as a simple wrapper in order to remain with the coding
        convention used throughout the rest of this project.

        '''

        self.format_data()
        self.mahalanobis_statistic(n_jobs=n_jobs)

        return self

def mahala_fcn(x,y):
    '''

    Parameters
    ----------

    x - numpy.ndarray
        A 1D array

    y - numpy.ndarray
        A 1D array

    '''

    cov = np.cov(zip(x,y))
    try:
        icov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        icov = np.linalg.inv(cov + np.eye(cov.shape[0], cov.shape[1], k=1e-3))

    diff = x - y
    val = np.dot(diff.T, np.dot(icov, diff))

    return np.sqrt(diff)

def intensity_data(cube, p=0.25):
    '''
    '''
    vec_length = round(p*cube.shape[1]*cube.shape[2])
    intensity_vecs = np.empty((cube.shape[0], vec_length))
    for dv in range(cube.shape[0]):
        vel_vec = cube[dv,:,:].ravel()
        vel_vec.sort()
        ## Return the normalized, shortened vector
        intensity_vecs[dv,:] = vel_vec[:vec_length]/np.max(vel_vec[:vec_length]) #

    return intensity_vecs