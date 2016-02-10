# Licensed under an MIT open source license - see LICENSE


import numpy as np

from ..threeD_to_twoD import var_cov_cube
from width_estimate import WidthEstimate1D, WidthEstimate2D


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

    def __init__(self, cube, n_eigs=None):
        super(PCA, self).__init__()
        self.cube = cube

        if n_eigs is None:
            self.n_eigs = self.cube.shape[0]
        else:
            self.n_eigs = n_eigs

        # Remove NaNs
        self.cube[np.isnan(self.cube)] = 0

        self.n_velchan = self.cube.shape[0]

    @property
    def cube(self):
        return self._cube

    @cube.setter
    def cube(self, input_cube):

        if len(input_cube.shape) > 3:
            input_cube = input_cube.squeeze()

        if len(input_cube.shape) != 3:
            raise ValueError("Must input a 3D cube "
                             "(velocity, position, position).")

        self._cube = input_cube

    @property
    def n_eigs(self):
        return self._n_eigs

    @n_eigs.setter
    def n_eigs(self, value):
        if value <= 0:
            raise ValueError("n_eigs must be > 0.")

        self._n_eigs = value

    def compute_pca(self, mean_sub=False, normalize=True):
        '''
        Create the covariance matrix and its eigenvalues.

        Parameters
        ----------
        normalize : bool, optional
            Normalize the set of eigenvalues by the 0th component.
        '''

        self.cov_matrix = var_cov_cube(self.cube, mean_sub=mean_sub)

        all_eigsvals, eigvecs = np.linalg.eig(self.cov_matrix)
        eigvecs = eigvecs[np.argsort(all_eigsvals)[::-1]]
        all_eigsvals = np.sort(all_eigsvals)[::-1]  # Sort by maximum

        self._var_prop = np.sum(all_eigsvals[:self.n_eigs]) / \
            np.sum(all_eigsvals)

        self._eigvecs = eigvecs[:, :self.n_eigs]

        if normalize:
            self._eigvals = all_eigsvals[:self.n_eigs] / all_eigsvals[0]
        else:
            self._eigvals = all_eigsvals[:self.n_eigs]

        return self

    @property
    def var_proportion(self):
        return self._var_prop

    @property
    def eigvals(self):
        return self._eigvals

    @property
    def eigvecs(self):
        return self._eigvecs

    def eigimages(self, n_eigs=None):

        if n_eigs is None:
            n_eigs = self.n_eigs

        if n_eigs > 0:
            iterat = xrange(n_eigs)
        elif n_eigs < 0:
            iterat = xrange(n_eigs, 0, 1)

        for ct, idx in enumerate(iterat):
            eigimg = np.zeros(self.cube.shape[1:], dtype=float)
            for channel in range(self.cube.shape[0]):
                eigimg += np.nan_to_num(self.cube[channel] *
                                        self.eigvecs[channel, idx])
            if ct == 0:
                eigimgs = eigimg
            else:
                eigimgs = np.dstack((eigimgs, eigimg))
        return eigimgs.swapaxes(0, 2)

    def autocorr_images(self, n_eigs=None):

        if n_eigs is None:
            n_eigs = self.n_eigs

        # Calculate the eigenimages
        eigimgs = self.eigimages(n_eigs=n_eigs)

        for idx, image in enumerate(eigimgs):
            fftx = np.fft.fft2(image)
            fftxs = np.conjugate(fftx)
            acor = np.fft.ifft2((fftx-fftx.mean())*(fftxs-fftxs.mean()))
            acor = np.fft.fftshift(acor)

            if idx == 0:
                acors = acor.real
            else:
                acors = np.dstack((acors, acor.real))

        return acors.swapaxes(0, 2)

    def autocorr_spec(self, n_eigs=None):

        if n_eigs is None:
            n_eigs = self.n_eigs

        for idx in range(n_eigs):
            fftx = np.fft.fft(self.eigvecs[:, idx])
            fftxs = np.conjugate(fftx)
            acor = np.fft.ifft((fftx-fftx.mean())*(fftxs-fftxs.mean()))
            if idx == 0:
                acors = acor.real
            else:
                acors = np.dstack((acors, acor.real))

        return acors.swapaxes(0, 1).squeeze()

    def noise_ACF(self, n_eigs=-10):

        if n_eigs is None:
            n_eigs = self.n_eigs

        acors = self.autocorr_images(n_eigs=n_eigs)

        noise_ACF = np.nansum(acors, axis=0) / float(n_eigs)

        return noise_ACF

    def find_spatial_widths(self, n_eigs=None, method='contour'):

        if n_eigs is None:
            n_eigs = self.n_eigs

        acors = self.autocorr_images(n_eigs=n_eigs)
        noise_ACF = self.noise_ACF()

        self._spatial_width, self._models = \
            WidthEstimate2D(acors, NoiseACF=noise_ACF, method=method)

    @property
    def spatial_width(self):
        return self._spatial_width

    def find_spectral_widths(self, n_eigs=None, method='interpolate'):
        '''
        Calculate the spectral scales for the structure functions.
        '''

        if n_eigs is None:
            n_eigs = self.n_eigs

        acorr_spec = self.autocorr_spec(n_eigs=n_eigs)

        self._spectral_width = WidthEstimate1D(acorr_spec, method=method)

    @property
    def spectral_width(self):
        return self._spectral_width

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
            p.bar(np.arange(1, self.n_eigs + 1), self.eigvals, 0.5, color='r')
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
            p.bar(np.arange(1, self.pca1.n_eigs + 1), self.pca1.eigvals, 0.5, color='r')
            p.xlim([0, self.pca1.n_eigs + 1])
            p.xlabel('Eigenvalues')
            p.ylabel('Variance')
            p.subplot(2, 2, 2)
            p.imshow(
                self.pca2.cov_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.title("PCA2")
            p.subplot(2, 2, 4)
            p.bar(np.arange(1, self.pca2.n_eigs + 1), self.pca2.eigvals, 0.5, color='r')
            p.xlim([0, self.pca2.n_eigs + 1])
            p.xlabel('Eigenvalues')

            p.tight_layout()
            p.show()

        return self
