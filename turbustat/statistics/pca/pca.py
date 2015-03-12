# Licensed under an MIT open source license - see LICENSE


import numpy as np


class PCA(object):

    '''
    Implementation of Principal Component Analysis (Heyer & Brunt, 2002)

    Parameters
    ----------

    cube : numpy.ndarray
        Data cube.
    n_eigs : int
        Number of eigenvalues to compute.
    '''

    def __init__(self, cube, n_eigs=50):
        super(PCA, self).__init__()
        self.cube = cube
        self.n_eigs = n_eigs

        # Remove NaNs
        self.cube[np.isnan(self.cube)] = 0

        self.n_velchan = self.cube.shape[0]
        self.eigvals = None

    def compute_pca(self, normalize=True):
        '''
        Create the covariance matrix and its eigenvalues.

        Parameters
        ----------
        normalize : bool, optional
            Normalize the set of eigenvalues by the 0th component.
        '''

        self.cov_matrix = np.zeros((self.n_velchan, self.n_velchan))

        for i, chan in enumerate(_iter_2D(self.cube)):
            norm_chan = chan - np.nanmean(chan)
            for j, chan2 in enumerate(_iter_2D(self.cube[:i+1, :, :])):
                norm_chan2 = chan2 - np.nanmean(chan2)

                self.cov_matrix[i, j] = \
                    np.nansum(norm_chan*norm_chan2) / \
                    (np.sum(np.isfinite(norm_chan*norm_chan2)) - 1)

        self.cov_matrix = self.cov_matrix + self.cov_matrix.T

        all_eigsvals, eigvecs = np.linalg.eig(self.cov_matrix)
        all_eigsvals = np.sort(all_eigsvals)[::-1]  # Sort by maximum

        self._var_prop = np.sum(all_eigsvals[:self.n_eigs]) / \
            np.sum(all_eigsvals)

        if normalize:
            self.eigvals = all_eigsvals[:self.n_eigs] / all_eigsvals[0]
        else:
            self.eigvals = all_eigsvals[:self.n_eigs]

        return self

    @property
    def var_proportion(self):
        return self._var_prop

    def run(self, verbose=False, normalize=True):
        '''
        Run method. Needed to maintain package standards.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        normalize : bool, optional
            See ```compute_pca```.
        '''

        self.compute_pca(normalize=normalize)

        if verbose:
            import matplotlib.pyplot as p

            print 'Proportion of Variance kept: %s' % (self.var_proportion)

            p.subplot(121)
            p.imshow(self.cov_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.subplot(122)
            p.bar(np.arange(1, self.n_eigs + 1, ), self.eigvals, 0.5, color='r')
            p.xlim([0, self.n_eigs + 1])
            p.xlabel('Eigenvalues')
            p.ylabel('Variance')
            p.show()


class PCA_Distance(object):

    '''
    Compare two data cubes based on the eigenvalues of the PCA decomposition.
    The distance is the Euclidean distance between the eigenvalues.

    Parameters
    ----------
    cube1 : numpy.ndarray
        Data cube.
    cube2 : numpy.ndarray
        Data cube.
    n_eigs : int
        Number of eigenvalues to compute.
    fiducial_model : PCA
        Computed PCA object. Use to avoid recomputing.
    normalize : bool, optional
        Sets whether to normalize the eigenvalues by the 0th eigenvalue.

    '''

    def __init__(self, cube1, cube2, n_eigs=50, fiducial_model=None,
                 normalize=True):
        super(PCA_Distance, self).__init__()
        self.cube1 = cube1
        self.cube2 = cube2

        if fiducial_model is not None:
            self.pca1 = fiducial_model
        else:
            self.pca1 = PCA(self.cube1, n_eigs=n_eigs)
            self.pca1.run(normalize=normalize)

        self.pca2 = PCA(self.cube2, n_eigs=n_eigs)
        self.pca2.run(normalize=normalize)

        self.distance = None

    def distance_metric(self, verbose=False):
        '''
        Computes the distance between the cubes.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        difference = np.abs((self.pca1.eigvals - self.pca2.eigvals) ** 2.)
        self.distance = np.sqrt(np.sum(difference))

        if verbose:
            import matplotlib.pyplot as p

            print "Proportions of total variance: 1 - %0.3f, 2 - %0.3f" % \
                (self.pca1.var_proportion, self.pca2.var_proportion)

            p.subplot(2, 2, 1)
            p.imshow(
                self.pca1.cov_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.title("PCA1")
            p.subplot(2, 2, 3)
            p.bar(np.arange(1, self.pca1.n_eigs + 1, ), self.pca1.eigvals, 0.5, color='r')
            p.xlim([0, self.pca1.n_eigs + 1])
            p.xlabel('Eigenvalues')
            p.ylabel('Variance')
            p.subplot(2, 2, 2)
            p.imshow(
                self.pca2.cov_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.title("PCA2")
            p.subplot(2, 2, 4)
            p.bar(np.arange(1, self.pca2.n_eigs + 1, ), self.pca2.eigvals, 0.5, color='r')
            p.xlim([0, self.pca2.n_eigs + 1])
            p.xlabel('Eigenvalues')

            p.tight_layout()
            p.show()

        return self


def _iter_2D(arr):
    '''
    Flatten a 3D cube into 2D by its channels.
    '''

    for chan in arr.reshape((arr.shape[0], -1)):
        yield chan
