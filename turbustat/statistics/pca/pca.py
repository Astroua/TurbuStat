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
        Number of eigenvalues to compute.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, n_eigs=50):
        super(PCA, self).__init__()

        # Header not needed
        self.need_header_flag = False
        self.header = None

        self.data = input_data(cube, no_header=True)
        self.n_eigs = n_eigs

        self.n_velchan = self.data.shape[0]
        self.eigvals = None

    def compute_pca(self, mean_sub=False, normalize=True):
        '''
        Create the covariance matrix and its eigenvalues.

        Parameters
        ----------
        mean_sub : bool, optional
            When enabled, subtracts the means of the channels before
            calculating the covariance. By default, this is disabled to
            match the Heyer & Brunt method.
        normalize : bool, optional
            Normalize the set of eigenvalues by the 0th component when mean
            subtraction has not been done. Otherwise this is normalized by the
            sum of the eigenvalues, so each represents the proportion of
            variance that eigenvector describes.
        '''

        self.cov_matrix = var_cov_cube(self.data, mean_sub=mean_sub)

        all_eigsvals, eigvecs = np.linalg.eig(self.cov_matrix)
        all_eigsvals = np.sort(all_eigsvals)[::-1]  # Sort by maximum

        self._var_prop = np.sum(all_eigsvals[:self.n_eigs]) / \
            np.sum(all_eigsvals)

        if normalize:
            if mean_sub:
                self.eigvals = all_eigsvals[:self.n_eigs] / \
                    np.sum(all_eigsvals[:self.n_eigs])
            else:
                self.eigvals = all_eigsvals[:self.n_eigs] / all_eigsvals[0]
        else:
            self.eigvals = all_eigsvals[:self.n_eigs]

    @property
    def var_proportion(self):
        return self._var_prop

    def run(self, verbose=False, normalize=True, mean_sub=False):
        '''
        Run method. Needed to maintain package standards.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        normalize : bool, optional
            See ```compute_pca```.
        '''

        self.compute_pca(normalize=normalize, mean_sub=mean_sub)

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
            if normalize:
                p.ylabel("Proportion of Variance")
            else:
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
    normalize : bool, optional
        Sets whether to normalize the eigenvalues by the 0th eigenvalue.
    mean_sub : bool, optional
        Subtracts the mean before computing the covariance matrix. The default
        is to not subtract the mean, as is done in the Heyer & Brunt works.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, n_eigs=50, fiducial_model=None,
                 normalize=True, mean_sub=False):
        super(PCA_Distance, self).__init__()

        self.normalize = normalize

        if fiducial_model is not None:
            self.pca1 = fiducial_model
        else:
            self.pca1 = PCA(cube1, n_eigs=n_eigs)
            self.pca1.run(normalize=normalize)

        self.pca2 = PCA(cube2, n_eigs=n_eigs)
        self.pca2.run(normalize=normalize)

    def distance_metric(self, verbose=False, label1=None, label2=None):
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

        difference = np.abs((self.pca1.eigvals - self.pca2.eigvals) ** 2.)
        self.distance = np.sqrt(np.sum(difference))

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
            p.bar(np.arange(1, self.pca1.n_eigs + 1), self.pca1.eigvals, 0.5,
                  color='r')
            p.xlim([0, self.pca1.n_eigs + 1])
            p.xlabel('Eigenvalues')
            if self.normalize:
                p.ylabel("Proportion of Variance")
            else:
                p.ylabel('Variance')
            p.subplot(2, 2, 2)
            p.imshow(
                self.pca2.cov_matrix, origin="lower", interpolation="nearest",
                vmin=np.median(self.pca2.cov_matrix))
            p.colorbar()
            p.title(label2)
            p.subplot(2, 2, 4)
            p.bar(np.arange(1, self.pca2.n_eigs + 1), self.pca2.eigvals, 0.5,
                  color='r')
            p.xlim([0, self.pca2.n_eigs + 1])
            p.xlabel('Eigenvalues')

            p.tight_layout()
            p.show()

        return self
