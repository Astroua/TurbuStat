# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

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

    def __init__(self, cube1, cube2, noise_value1=-np.inf,
                 noise_value2=-np.inf):
        super(Cramer_Distance, self).__init__()
        self.cube1 = input_data(cube1, no_header=True)
        self.cube2 = input_data(cube2, no_header=True)

        self.noise_value1 = noise_value1
        self.noise_value2 = noise_value2

    @property
    def data_matrix1(self):
        '''
        2D representation of `cube1`. Each column contains the
        brightest N pixels in a spectral channel, set in
        `~Cramer_Distance.format_data`.
        '''
        return self._data_matrix1

    @property
    def data_matrix2(self):
        '''
        2D representation of `cube2`. Each column contains the
        brightest N pixels in a spectral channel, set in
        `~Cramer_Distance.format_data`.
        '''
        return self._data_matrix2

    def format_data(self, data_format='intensity', seed=13024, normalize=True,
                    **kwargs):
        '''
        Rearrange data into a 2D object using the given format.

        Parameters
        ----------
        data_format : {'intensity', 'spectra'}, optional
            The method to use to construct the data matrix. The default is
            intensity, which picks the brightest values in each channel. The
            other option is 'spectra', which will pick the N brightest spectra
            to compare.
        seed : int, optional
            When the data are mismatched, the larger data set is randomly
            sampled to match the size of the other.
        normalize : bool, optional
            Forces the data sets into the same interval, removing the
            effect of different ranges of intensities (or whatever unit the
            data traces).
        kwargs : Passed to `~turbustat.statistics.threeD_to_twoD._format_data`.
        '''

        self._data_matrix1 = _format_data(self.cube1, data_format=data_format,
                                          noise_lim=self.noise_value1,
                                          normalize=normalize, **kwargs)
        self._data_matrix2 = _format_data(self.cube2, data_format=data_format,
                                          noise_lim=self.noise_value2,
                                          normalize=normalize, **kwargs)

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

                self._data_matrix2 = new_data

            else:

                new_data = np.empty((self.data_matrix1.shape[0], samps2))

                for i in range(self.data_matrix1.shape[0]):
                    new_data[i, :] = \
                        np.random.choice(self.data_matrix1[i, :], samps2,
                                         replace=False)

                self._data_matrix1 = new_data

    def cramer_statistic(self, n_jobs=1):
        '''
        Applies the Cramer Statistic to the datasets.

        Parameters
        ----------
        n_jobs : int, optional
            Sets the number of cores to use to calculate
            pairwise distances. Default is 1.
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

        pairdist11 = pairwise_distances(larger, metric="euclidean",
                                        n_jobs=n_jobs)
        pairdist22 = pairwise_distances(smaller, metric="euclidean",
                                        n_jobs=n_jobs)
        pairdist12 = pairwise_distances(larger, smaller,
                                        metric="euclidean", n_jobs=n_jobs)

        # Take sqrt of each
        # We default to using the Cramer kernel in Baringhaus & Franz (2004)
        # \phi(dist) = sqrt(dist) / 2.
        # The normalization values below reflect this
        pairdist11 = np.sqrt(pairdist11)
        pairdist12 = np.sqrt(pairdist12)
        pairdist22 = np.sqrt(pairdist22)

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

        self._distance = (m * n / (m + n)) * (term1 - term2 - term3)

    @property
    def distance(self):
        '''
        Cramer distance between `cube1` and `cube2`.
        '''
        return self._distance

    def distance_metric(self, verbose=False, normalize=True, n_jobs=1,
                        label1="1", label2="2", save_name=None):
        '''

        Run the Cramer statistic.

        Parameters
        ----------
        verbose : bool, optional
            Enable plotting of the data matrices.
        normalize : bool, optional
            See `Cramer_Distance.format_data`.
        n_jobs : int, optional
            See `Cramer_Distance.cramer_statistic`.
        label1 : str, optional
            Object or region name for data1
        label2 : str, optional
            Object or region name for data2
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        self.format_data(normalize=normalize)
        self.cramer_statistic(n_jobs=n_jobs)

        if verbose:

            import matplotlib.pyplot as plt

            all_max = max(self.data_matrix1.max(),
                          self.data_matrix2.max())
            all_min = min(self.data_matrix1.min(),
                          self.data_matrix2.min())

            plt.subplot(121)
            plt.title(label1)
            plt.imshow(self.data_matrix1.T, origin='lower',
                       vmin=all_min, vmax=all_max)
            plt.yticks([])
            plt.xticks([0, self.data_matrix1.shape[0]])
            plt.xlabel("Channel")

            plt.subplot(122)
            plt.title(label2)
            plt.imshow(self.data_matrix2.T, origin='lower',
                       vmin=all_min, vmax=all_max)
            plt.colorbar()
            plt.yticks([])
            plt.xticks([0, self.data_matrix2.shape[0]])
            plt.xlabel("Channel")

            plt.tight_layout()

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self
