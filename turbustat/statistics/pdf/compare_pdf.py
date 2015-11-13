# Licensed under an MIT open source license - see LICENSE

'''

The density PDF as described by Kowal et al. (2007)

'''

import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp
import warnings

from ..stats_utils import hellinger, standardize, common_histogram_bins


class PDF(object):
    '''
    Create the PDF of a given array.

    Parameters
    ----------
    img : numpy.ndarray
        A 1-3D array.
    min_val : float, optional
        Minimum value to keep in the given image.
    bins : list or numpy.ndarray or int, optional
        Bins to compute the PDF from.
    weights : numpy.ndarray, optional
        Weights to apply to the image. Must have the same shape as the image.
    use_standardized : bool, optional
        Enable to standardize the data before computing the PDF and ECDF.
    '''
    def __init__(self, img, min_val=0.0, bins=None, weights=None,
                 use_standardized=False):
        super(PDF, self).__init__()

        self.img = img

        # We want to remove NaNs and value below the threshold.
        self.data = img[np.isfinite(img)]
        self.data = self.data[self.data > min_val]

        # Do the same for the weights, then apply weights to the data.
        if weights is not None:
            self.weights = weights[np.isfinite(img)]
            self.weights = self.weights[self.data > min_val]

            self.data *= self.weights

        self._standardize_flag = False
        if use_standardized:
            self._standardize_flag = True
            self.data = standardize(self.data)

        self._bins = bins

    def make_pdf(self, bins=None):
        '''
        Create the PDF.

        Parameters
        ----------
        bins : list or numpy.ndarray or int, optional
            Bins to compute the PDF from. Overrides initial bin input.
        '''

        if bins is not None:
            self._bins = bins

        # If the number of bins is not given, use sqrt of data length.
        if self.bins is None:
            self._bins = np.sqrt(self.data.shape[0])
            self._bins = int(np.round(self.bins))

        norm_weights = np.ones_like(self.data) / self.data.shape[0]

        self._pdf, bin_edges = np.histogram(self.data, bins=self.bins,
                                            density=False,
                                            weights=norm_weights)

        self._bins = (bin_edges[:-1] + bin_edges[1:]) / 2

        return self

    @property
    def pdf(self):
        return self._pdf

    @property
    def bins(self):
        return self._bins

    def make_ecdf(self):
        '''
        Create the ECDF.
        '''

        self._ecdf = np.cumsum(np.sort(self.data.ravel())) / np.sum(self.data)

        return self

    @property
    def ecdf(self):
        return self._ecdf

    def run(self, verbose=False, bins=None):
        '''
        Compute the PDF and ECDF. Enabling verbose provides
        a summary plot.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting of the results.
        bins : list or numpy.ndarray or int, optional
            Bins to compute the PDF from. Overrides initial bin input.
        '''

        self.make_pdf(bins=bins)
        self.make_ecdf()

        if verbose:

            if self._standardize_flag:
                xlabel = r"z-score"
            else:
                xlabel = r"$\Sigma$"

            import matplotlib.pyplot as p
            # PDF
            p.subplot(131)
            p.loglog(self.bins[self.pdf > 0], self.pdf[self.pdf > 0], 'b-')
            p.grid(True)
            p.xlabel(xlabel)
            p.ylabel("PDF")

            # ECDF
            p.subplot(132)
            p.semilogx(np.sort(self.data.ravel()), self.ecdf, 'b-')
            p.grid(True)
            p.xlabel(xlabel)
            p.ylabel("ECDF")

            # Array representation.
            p.subplot(133)
            if self.img.ndim == 1:
                p.plot(self.img, 'b-')
            elif self.img.ndim == 2:
                p.imshow(self.img, origin="lower", interpolation="nearest",
                         cmap="binary")
            elif self.img.ndim == 3:
                p.imshow(np.nansum(self.img, axis=0), origin="lower",
                         interpolation="nearest", cmap="binary")
            else:
                print("Visual representation works only up to 3D.")

            p.show()

        return self


class PDF_Distance(object):
    '''
    Calculate the distance between two arrays using their PDFs.

    Parameters
    ----------
    img1 : numpy.ndarray
        Array (1-3D).
    img2 : numpy.ndarray
        Array (1-3D).
    min_val1 : float, optional
        Minimum value to keep in img1
    min_val2 : float, optional
        Minimum value to keep in img2
    weights1 : numpy.ndarray, optional
        Weights to be used with img1
    weights2 : numpy.ndarray, optional
        Weights to be used with img2
    '''
    def __init__(self, img1, img2, min_val1=0.0, min_val2=0.0,
                 weights1=None, weights2=None):
        super(PDF_Distance, self).__init__()

        self.img1 = img1
        self.img2 = img2

        self.PDF1 = PDF(self.img1, min_val=min_val1, use_standardized=True,
                        weights=weights1)

        self.PDF2 = PDF(self.img2, min_val=min_val2, use_standardized=True,
                        weights=weights2)

        self.bins, self.bin_centers = \
            common_histogram_bins(self.PDF1.data, self.PDF2.data,
                                  return_centered=True)

        # Feed the common set of bins to be used in the PDFs
        self.PDF1.run(verbose=False, bins=self.bins)
        self.PDF2.run(verbose=False, bins=self.bins)

    def compute_hellinger_distance(self):
        '''
        Computes the Hellinger Distance between the two PDFs.
        '''

        self.hellinger_distance = hellinger(self.PDF1.pdf, self.PDF2.pdf)

        return self

    def compute_ks_distance(self):
        '''
        Compute the distance using the KS Test.
        '''

        D, p = ks_2samp(self.PDF1.data, self.PDF2.data)

        self.ks_distance = D
        self.ks_pval = p

        return self

    def compute_ad_distance(self):
        '''
        Compute the distance using the Anderson-Darling Test.
        '''

        raise NotImplementedError(
            "Use of the Anderson-Darling test has been disabled"
            " due to occurence of overflow errors.")

        # D, _, p = anderson_ksamp([self.PDF1.data, self.PDF2.data])

        # self.ad_distance = D
        # self.ad_pval = p

    def distance_metric(self, statistic='all', labels=None, verbose=False):
        '''
        Calculate the distance.
        *NOTE:* The data are standardized before comparing to ensure the
        distance is calculated on the same scales.

        Parameters
        ----------
        statistic : 'all', 'hellinger', 'ks'
            Which measure of distance to use.
        labels : tuple, optional
            Sets the labels in the output plot.
        verbose : bool, optional
            Enables plotting.
        '''

        if statistic is 'all':
            self.compute_hellinger_distance()
            self.compute_ks_distance()
            # self.compute_ad_distance()
        elif statistic is 'hellinger':
            self.compute_hellinger_distance()
        elif statistic is 'ks':
            self.compute_ks_distance()
        # elif statistic is 'ad':
        #     self.compute_ad_distance()
        else:
            raise TypeError("statistic must be 'all',"
                            "'hellinger', or 'ks'.")
                            # "'hellinger', 'ks' or 'ad'.")

        if verbose:
            import matplotlib.pyplot as p
            # PDF
            if labels is None:
                label1 = "Input 1"
                label2 = "Input 2"
            else:
                label1 = labels[0]
                label2 = labels[1]
            p.subplot(131)
            p.loglog(self.bin_centers[self.PDF1.pdf > 0],
                     self.PDF1.pdf[self.PDF1.pdf > 0], 'b-', label=label1)
            p.loglog(self.bin_centers[self.PDF2.pdf > 0],
                     self.PDF2.pdf[self.PDF2.pdf > 0], 'g-', label=label2)
            p.legend(loc="best")
            p.grid(True)
            p.xlabel(r"z-score")
            p.ylabel("PDF")

            # ECDF
            p.subplot(132)
            p.semilogx(np.sort(self.PDF1.data.ravel()), self.PDF1.ecdf, 'b-')
            p.semilogx(np.sort(self.PDF2.data.ravel()), self.PDF2.ecdf, 'g-')
            p.grid(True)
            p.xlabel(r"z-score")
            p.ylabel("ECDF")

            # Array representation.
            p.subplot(233)
            if self.img1.ndim == 1:
                p.plot(self.img1, 'b-')
            elif self.img1.ndim == 2:
                p.imshow(self.img1, origin="lower", interpolation="nearest",
                         cmap="binary")
            elif self.img1.ndim == 3:
                p.imshow(np.nansum(self.img1, axis=0), origin="lower",
                         interpolation="nearest", cmap="binary")
            else:
                print("Visual representation works only up to 3D.")

            p.subplot(236)
            if self.img2.ndim == 1:
                p.plot(self.img2, 'b-')
            elif self.img2.ndim == 2:
                p.imshow(self.img2, origin="lower", interpolation="nearest",
                         cmap="binary")
            elif self.img2.ndim == 3:
                p.imshow(np.nansum(self.img2, axis=0), origin="lower",
                         interpolation="nearest", cmap="binary")
            else:
                print("Visual representation works only up to 3D.")

            p.show()

        return self
