# Licensed under an MIT open source license - see LICENSE

'''

The density PDF as described by Kowal et al. (2007)

'''

import numpy as np


class PDF(object):
    '''
    Create the PDF of a given array.
    '''
    def __init__(self, img, min_val=0.0, bins=None, weights=None, norm=False):
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

        if norm:
            # Normalize by the average
            self.data /= np.mean(self.data, axis=None)

        self._bins = bins

    def make_pdf(self, bins=None):
        '''
        Create the PDF.
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

        self._ecdf = np.cumsum(self.pdf)

        return self

    @property
    def ecdf(self):
        return self._ecdf

    def run(self, verbose=False):
        '''
        Run the whole thing.
        '''

        self.make_pdf()
        self.make_ecdf()

        if verbose:
            import matplotlib.pyplot as p
            # PDF
            p.subplot(131)
            p.loglog(self.bins[self.pdf > 0], self.pdf[self.pdf > 0], 'b-')
            p.grid(True)
            p.xlabel(r"$\Sigma/\overline{\Sigma}$")
            p.ylabel("PDF")

            # ECDF
            p.subplot(132)
            p.semilogx(self.bins, self.ecdf, 'b-')
            p.grid(True)
            p.xlabel(r"$\Sigma/\overline{\Sigma}$")
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

        if weights1 is None:
            weights1 = np.ones(img1.shape)

        if weights2 is None:
            weights2 = np.ones(img2.shape)

        # We want to make sure we're using the same set of bins for the
        # comparisons. Unfortunately, we have redundant calculations to
        # do this, but it is somewhat necessary to keep PDF standalone.

        stand1 = standardize((img1 * weights1)[np.isfinite(img1) |
                                               (img1 > min_val1)])
        stand2 = standardize((img2 * weights2)[np.isfinite(img2) |
                                               (img2 > min_val2)])

        max_val = max(np.nanmax(stand1),
                      np.nanmax(stand2))
        min_val = min(np.nanmin(stand1),
                      np.nanmin(stand2))

        # Number of bins is the sqrt of the average between the number of
        # good values.
        num_bins = (stand1.shape[0] +
                    stand2.shape[0]) / 2

        num_bins = int(np.round(np.sqrt(num_bins)))

        self.bins = np.linspace(min_val, max_val, num_bins)

        self.PDF1 = PDF(stand1, bins=self.bins)
        self.PDF1.run(verbose=False)

        self.PDF2 = PDF(stand2, bins=self.bins)
        self.PDF2.run(verbose=False)

    def distance_metric(self, verbose=False):
        '''
        Calculate the distance.
        *NOTE:* The data are standardized before comparing to ensure the
        distance is calculated on the same scales.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        stand_data1 = standardize(self.PDF1.pdf)
        stand_data2 = standardize(self.PDF2.pdf)

        # Now that its standardized, calculate the distance.

        self.distance = hellinger(stand_data1, stand_data2)

        if verbose:
            import matplotlib.pyplot as p
            # PDF
            p.subplot(131)
            p.loglog(self.PDF1.bins[self.PDF1.pdf > 0],
                     self.PDF1.pdf[self.PDF1.pdf > 0], 'b-', label="Input 1")
            p.loglog(self.PDF1.bins[self.PDF2.pdf > 0],
                     self.PDF2.pdf[self.PDF2.pdf > 0], 'g-', label="Input 2")
            p.legend(loc="best")
            p.grid(True)
            p.xlabel(r"$\Sigma/\overline{\Sigma}$")
            p.ylabel("PDF")

            # ECDF
            p.subplot(132)
            p.semilogx(self.PDF1.bins, self.PDF1.ecdf, 'b-')
            p.semilogx(self.PDF2.bins, self.PDF2.ecdf, 'g-')
            p.grid(True)
            p.xlabel(r"$\Sigma/\overline{\Sigma}$")
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


def hellinger(data1, data2):
    '''
    Calculate the Hellinger Distance between two datasets.

    Parameters
    ----------
    data1 : numpy.ndarray
        1D array.
    data2 : numpy.ndarray
        1D array.

    Returns
    -------
    distance : float
        Distance value.
    '''
    distance = (1 / np.sqrt(2)) * \
        np.sqrt(np.nansum((np.sqrt(data1) - np.sqrt(data2)) ** 2.))
    return distance


def standardize(x):
    return (x - np.nanmean(x)) / np.nanstd(x)
