# Licensed under an MIT open source license - see LICENSE


import numpy as np

from ..threeD_to_twoD import var_cov_cube
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types, input_data


class PCA(BaseStatisticMixIn):

    '''
    Implementation of Principal Component Analysis (Heyer & Brunt, 2002)

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    n_eigs : int
        Number of eigenvalues to compute. Defaults to using all eigenvalues.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, n_eigs=-1):
        super(PCA, self).__init__()

        # Header not needed
        self.need_header_flag = False
        self.header = None

        self.data = input_data(cube, no_header=True)

        self.spectral_shape = self.data.shape[0]

        if n_eigs == -1:
            self.n_eigs = self.spectral_shape
        elif n_eigs < -1 or n_eigs > self.spectral_shape or n_eigs == 0:
            raise Warning("n_eigs must be less than the number of velocity"
                          " channels ({}) or -1 for"
                          " all".format(self.spectral_shape))
        else:
            self.n_eigs = n_eigs

    def compute_pca(self, mean_sub=False):
        '''
        Create the covariance matrix and its eigenvalues.

        If `mean_sub` is disabled, the first eigenvalue is dominated by the
        mean of the data, not the variance.

        Parameters
        ----------
        mean_sub : bool, optional
            When enabled, subtracts the means of the channels before
            calculating the covariance. By default, this is disabled to
            match the Heyer & Brunt method.
        '''

        self.cov_matrix = var_cov_cube(self.data, mean_sub=mean_sub)

        all_eigsvals, eigvecs = np.linalg.eig(self.cov_matrix)
        all_eigsvals = np.sort(all_eigsvals)[::-1]  # Sort by maximum

        if mean_sub:
            self._total_variance = np.sum(all_eigsvals)
            self._var_prop = np.sum(all_eigsvals[:self.n_eigs]) / \
                self.total_variance
        else:
            self._total_variance = np.sum(all_eigsvals[1:])
            self._var_prop = np.sum(all_eigsvals[1:self.n_eigs]) / \
                self.total_variance

        self.eigvals = all_eigsvals[:self.n_eigs]

    @property
    def var_proportion(self):
        return self._var_prop

    @property
    def total_variance(self):
        return self._total_variance

    def run(self, verbose=False, mean_sub=False):
        '''
        Run method. Needed to maintain package standards.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        self.compute_pca(mean_sub=mean_sub)

        if verbose:
            import matplotlib.pyplot as p

            print 'Proportion of Variance kept: %s' % (self.var_proportion)

            p.subplot(121)
            p.imshow(self.cov_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.subplot(122)
            p.bar(np.arange(1, self.n_eigs + 1), self.eigvals, 0.5, color='r')
            p.xlim([0, self.n_eigs + 1])
            p.xlabel('Eigenvalues')
            p.ylabel('Variance')
            p.show()

        return self


class PCA_Distance(object):

    '''
    Compare two data cubes based on the eigenvalues of the PCA decomposition.
    The distance is the Euclidean distance between the eigenvalues.

    Parameters
    ----------
    cube1 : %(dtypes)s
        Data cube.
    cube2 : %(dtypes)s
        Data cube.
    n_eigs : int
        Number of eigenvalues to compute.
    fiducial_model : PCA
        Computed PCA object. Use to avoid recomputing.
    mean_sub : bool, optional
        Subtracts the mean before computing the covariance matrix. Not
        subtracting the mean is done in the original Heyer & Brunt works.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, n_eigs=50, fiducial_model=None,
                 mean_sub=True):
        super(PCA_Distance, self).__init__()

        if fiducial_model is not None:
            self.pca1 = fiducial_model
        else:
            self.pca1 = PCA(cube1, n_eigs=n_eigs)
            self.pca1.run(mean_sub=mean_sub)

        self.pca2 = PCA(cube2, n_eigs=n_eigs)
        self.pca2.run(mean_sub=mean_sub)

        self._mean_sub = mean_sub

    def distance_metric(self, verbose=False, label1="Cube 1", label2="Cube 2"):
        '''
        Computes the distance between the cubes.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for cube1
        label2 : str, optional
            Object or region name for cube2
        '''

        # The eigenvalues need to be normalized before being compared. If
        # mean_sub is False, the first eigenvalue is not used.
        if self._mean_sub:
            eigvals1 = self.pca1.eigvals / np.sum(self.pca1.eigvals)
            eigvals2 = self.pca2.eigvals / np.sum(self.pca2.eigvals)
        else:
            eigvals1 = self.pca1.eigvals[1:] / np.sum(self.pca1.eigvals[1:])
            eigvals2 = self.pca2.eigvals[1:] / np.sum(self.pca2.eigvals[1:])

        self.distance = np.linalg.norm(eigvals1 - eigvals2)

        if verbose:
            import matplotlib.pyplot as p

            print "Proportions of total variance: 1 - %0.3f, 2 - %0.3f" % \
                (self.pca1.var_proportion, self.pca2.var_proportion)

            p.subplot(2, 2, 1)
            p.imshow(
                self.pca1.cov_matrix, origin="lower", interpolation="nearest",
                vmin=np.median(self.pca1.cov_matrix))
            p.colorbar()
            p.title(label1)
            p.subplot(2, 2, 3)
            p.bar(np.arange(1, len(eigvals1) + 1), eigvals1, 0.5,
                  color='r')
            p.xlim([0, self.pca1.n_eigs + 1])
            p.xlabel('Eigenvalues')
            p.ylabel("Proportion of Variance")
            p.subplot(2, 2, 2)
            p.imshow(
                self.pca2.cov_matrix, origin="lower", interpolation="nearest",
                vmin=np.median(self.pca2.cov_matrix))
            p.colorbar()
            p.title(label2)
            p.subplot(2, 2, 4)
            p.bar(np.arange(1, len(eigvals2) + 1), eigvals2, 0.5,
                  color='r')
            p.xlim([0, self.pca2.n_eigs + 1])
            p.xlabel('Eigenvalues')

            p.tight_layout()
            p.show()

        return self
