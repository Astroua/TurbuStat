
'''

Implementation of Principal Component Analysis (Heyer & Brunt, 2002)

'''

import numpy as np

class PCA(object):
    """docstring for PCA"""
    def __init__(self, cube, n_eigs=50):
        super(PCA, self).__init__()
        self.cube = cube
        self.n_eigs = n_eigs

        self.cube[np.isnan(self.cube)] = 0

        self.n_velchan = self.cube.shape[0]
        self.pca_matrix = np.zeros((self.n_velchan,self.n_velchan))
        self.eigvals = None

    def compute_pca(self):

        cube_mean = np.nansum(self.cube)/np.sum(np.isfinite(self.cube))
        norm_cube = self.cube - cube_mean

        for i in range(self.n_velchan):
            for j in range(i):
                self.pca_matrix[i,j] = np.nansum(norm_cube[i,:,:]*norm_cube[j,:,:])/\
                np.sum(np.isfinite(norm_cube[i,:,:]*norm_cube[j,:,:]))
        self.pca_matrix = self.pca_matrix + self.pca_matrix.T

        all_eigsvals, eigvecs = np.linalg.eig(self.pca_matrix)
        all_eigsvals.sort() # Sort by maximum
        self.eigvals = all_eigsvals[:self.n_eigs]

        return self

    def run(self, verbose=False):

        self.compute_pca()

        if verbose:
            import matplotlib.pyplot as p

            p.imshow(self.pca_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.show()

class PCA_Distance(object):
    """docstring for PCA_Distance"""
    def __init__(self, cube1, cube2, n_eigs=50, fiducial_model=None):
        super(PCA_Distance, self).__init__()
        self.cube1 = cube1
        self.cube2 = cube2

        if fiducial_model is not None:
            self.pca1 = fiducial_model
        else:
            self.pca1 = PCA(cube1, n_eigs=n_eigs)
            self.pca1.run()

        self.pca2 = PCA(cube2, n_eigs=n_eigs)
        self.pca2.run()

        self.distance = None

    def distance_metric(self, verbose=False):

        difference = np.abs((self.pca1.eigvals - self.pca2.eigvals)**2.)
        self.distance = np.sqrt(np.sum(difference))

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(1,2,1)
            p.imshow(self.pca1.pca_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.title("PCA1")

            p.subplot(1,2,2)
            p.imshow(self.pca2.pca_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.title("PCA2")

            p.show()

        return self